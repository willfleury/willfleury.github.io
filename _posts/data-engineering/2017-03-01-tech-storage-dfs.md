---
layout: default
title: Technology, Storage and Data Format Choices
description: What are the technology choices when implementing the Lambda Architecture
categories: [data engineering, boxever]
---

## Technology, Storage and Data Format Choices

This post is part of a [Series]({% post_url /data-engineering/2017-03-01-overview %}) on the Lambda Architecture.

### Overview

As we discussed in our previous post outlining this [series]({% post_url /data-engineering/2017-03-01-overview %}), when we started out on this journey we had no batch data pipeline at all and everything was stored in a single monolithic Cassandra database. This meant that there were a huge number of unknowns which we had to solve. Simple questions were unanswerable such as how many entities of type A did we have (remember that we couldn’t read all of our data out of the monolith cluster)? What size would all our data be in format X or Y? How long would it take to rebuild view Z? What was the best framework? Would our bottleneck be storage speed, network, cpu or memory? It's difficult to plan too far ahead when you cannot answer these questions and so as we progressed through each stage in the pipeline we had to analyse what we learned from the previous stages and see if it had any impact on how we thought we would approach the subsequent stage.

We had a few months earlier developed a Batch Import product feature which allowed us to familiarise ourselves with the myriad of technology choices available in this space (albeit for a different use case). There are a number of initial choices you have to make which can become important later. All choices can be changed later of course - the more you learn or as spaces evolve - but it is important to weigh up your options correctly. 

The main initial choices were we had to make were:

* File Storage system
    * HDFS or S3
* Append only distributed messaging system
    * Kafka
* Data processing framework
    * Mesos, EMR, Map-Reduce (YARN), Spark, Flink, Cascading, Scalding 
* Workflow Management and Scheduler
    * Oozie, Chronos, Azkaban, AWS Data Pipeline, Luigi, etc
* Data Formats
    * Row Formats - Json, Avro, Thrift, etc
    * Column Format - Parquet, ORC, etc

We will not provide an exhaustive description of each or why we chose each but we will attempt to communicate our key reasons briefly. Two options which we left to make until we progressed further were the serving datastores for the speed view and for the read-only batch view. 

### Storage System

When it comes to distributed, highly available file based storage systems you really only have two choices - HDFS or S3. HDFS requires that you run and manage a highly available (multi-zone) Hadoop cluster full time. We run everything on AWS and EMR is the AWS hosted Hadoop offering. However EMR runs in a single zone for performance reasons which means it is not highly available and cannot be used as a highly available datastore. Managing your own cluster is expensive and requires real operational experience with Hadoop (which we did not have at the time). In addition to this with a single HDFS cluster you end up having to worry about jobs competing for resources, disk throughput and network bandwidth and sometimes you end up with multiple clusters because of this. However the transfer of data between clusters can be slow. You also need to manage the scaling of this cluster as your data sizes grow. 

Given we were already working within AWS, S3 was another option. With S3 we did not have to worry about managing a cluster ourselves. We did not have to think about the scaling issues associated with this and could plan our costs relatively easily around this. We would also get eleven nines durability. Finally, we could have many jobs reading and writing to the same or different buckets at the same time without any impact in performance or cross job slowdown. Indeed, we have found that working with S3 the network bandwidth is typically our limiting factor! As an added benefit S3 provides versioning and lifecycle management which means you’re covered from accidental data removal or corruption. S3 is also an order of magnitude cheaper for us than having a dedicated HDFS cluster and it also integrates seamlessly with Glacier for longer and cheaper storage of data

Sometimes people worry about S3 consistency guarantees and some of the issues around availability and performance it had in the early days. As already mentioned the performance concerns are no longer valid and as S3 provides strong read-after-write consistency for PUTs of new objects consistency is not an issue either (be aware though for PUTs of existing objects or DELETEs - see the AWS faq on the matter [here](https://aws.amazon.com/s3/faqs/) although if accessing via EMR you have the option to use [EMRFS](http://docs.aws.amazon.com/emr/latest/ManagementGuide/emr-fs.html) for consistency even then). 

### Data Processing Framework(s)

In some ways choosing EMR was easy. We wanted to be able to take advantage of the myriad existing big data tools (s3distcp etc) which only run on YARN. EMR is a really powerful hosted Hadoop environment. It works well with S3 and integrates tightly with auto scaling features in AWS and the spot instance market. This meant that we could easily scale up our computing power whenever we needed and more importantly scale it down when no longer required it. An alternative to this would be to manage that ourselves or to pick a solution like Mesos. However we did not have an existing Mesos cluster and we did not have the time, resources or expertise to look at implementing this and working through a lot of the unknowns. Hence EMR it was! There were various challenges to configuring EMR to meet our requirements - especially around disk encryption which took some time to sort out (the EMR team were very helpful and disk encryption support is now a configuration option with EMR 5.x). We also experienced severe indigestion when upgrading from EMR 3.x to 4.x and 5.x.. The upgrade to a newer version is usually worth it but there is no free lunch! 

In terms of distributed frameworks to use for our ETL and analytics jobs we investigated a number of options that would be able to run on YARN (in fact, the all support YARN). We looked at the old Map-Reduce and libraries which make working with this easier such as Scalding or Cascading. While excellent libraries, we felt that it would be wrong to choose them given the new breed of distributed engines coming out at the time such as Spark and Flink. 

Flink is a really awesome engine and its approach to processing data (pipelined execution) is much better than Spark's (staged execution). Similarly Flink wins the stream processing competition hands down even though we were not looking to use it for streaming solutions straight away. However Flink was only entering Apache incubator status and was still rough around the edges when we started on this road. It also did not have as vibrant a community and ecosystem built around it (and still does not although it is improving rapidly). We decided to go with Spark which at the time was on version 1.2. We have not regretted this decision and while we are looking to use Flink for certain use cases now, Spark has served us well. However as anyone who has worked with Spark knows: again, nothing comes for free! 

At the time we chose to run Spark on EMR, even the Databricks EC2 scripts provided with it worked by launching Spark as a standalone cluster on regular EC2 instances. It wasn’t supported as an EMR application out of the box and we had to implement it ourselves as a bootstrap script on EMR. However we felt that running Spark on YARN was the best choice for running clustered Spark outside of Mesos. We believed that this choice was the correct one and soon after, were vindicated when AWS started providing Spark as an out of the box application for EMR. 

Today, Apache Flink is still on our radar and is a personal favourite. Note we have been using Apache Storm for some specific streaming tasks for the last number of years and while it has also served us well we would not choose it for any new tasks. Apache Flink provides the ability to run Storm topologies which makes it all the more appealing to start concentrating on it as a replacement.

### Workflow Management & Scheduler

There are many workflow schedulers available out there now that are actively maintained and more than a handful are part of the Apache project. However, when you start peeling back the layers of each solution, only one solution really aligned with the type of workflows, scheduling and technology requirements we had (at least initially). Our requirements centered around:

* Scheduled & Ad-hoc Jobs
* EMR (i.e. YARN Compatible)
* Tight Spark Integration but ability to run other Job types easily
* Error Handling
* Data Dependencies
* REST API
* CLI

As mentioned we were going to be running primarily Spark jobs on EMR. Hence we needed a workflow manager and scheduler which had tight integration and support with YARN. [Chronos](https://mesos.github.io/chronos/) has good EMR integration but it runs on Mesos. We did not want to have to host another set of nodes which we could launch the tasks on either and so the scheduled tasks must themselves run on YARN. [Azkaban](https://azkaban.github.io/) launches tasks on its own cluster and so you must setup an Azkaban cluster to run jobs which is yet more management we did not want to have to tackle at the time. 

Lastly we would have dependent jobs in our workflows. To manage those dependencies, we wanted a workflow manager that would ensure that we could define dependencies between jobs based on the presence (or absence) of data on S3. Of course you can write this yourself as a custom task but with [Apache Oozie](http://oozie.apache.org/) it came out of the box in the form of data dependencies. In terms of features we required Oozie won hands down. It feels a bit like the previous generation of tools but it does what we need and it does it well. 

A Spark Action had just been released with the 4.2.0 release of Oozie which also lined up nicely and made our lives that little bit easier (previously we had to use a Java action which wasn’t as smooth). 

As with our choice of Spark on YARN on EMR, our choice of Oozie on EMR also proved to be slightly ahead of the curve in terms of AWS offerings and as of EMR 5.x Oozie is now a fully supported out-of-the-box EMR Application. 

We have to caveat all of the above with the knowledge and hindsight of how [AWS Data Pipeline](https://aws.amazon.com/datapipeline/) has progressed in the last year and a half. If we were choosing it again we would probably choose AWS Data Pipeline as it fulfills all of our needs with Oozie and has the added benefit of being independent of any EMR cluster or anything we host ourselves. It also has the enormous advantage of being able to create EMR clusters itself as you need them to run a given workflow. This takes a lot of hassle and isolation concerns out of the question completely as you can run large workflows independently in their own clusters. This also aligns with our AWS first principles.

### Data Format

If you care about the environment you won’t use JSON as your primary data storage format. Data sizes are bloated and performance of reading and writing is expensive compared to more efficient binary schema based formats. It can also result in very poor data quality as given its relatively schemaless nature and that validation at the point of capture is not always guaranteed as it is not forced. 

There are many benefits to having a schema based format such as Avro. It ensures better validation, more evolvable schemas with backward and forward compatibility guaranteed by the schema itself. Schema based formats also naturally have more efficient wire size and are faster to read and write. 

Having said all this, Json was thoroughly rooted into our platform. It wasn’t something we could easily switch out to Avro or any other schema based format. We felt (and still do) that this is a separate project in itself and to reduce any risks to our primary goal we decided to stick with Json. Json is also easy to read and debug and to replace it with a binary format you would need to ensure the adequate tooling is in place to not hinder development and debugging (again, it's a separate project). [Confluent](https://www.confluent.io/) is definitely a solution we would look at to help around avro usage. It has some really nice services and tools for working with Avro and Kafka.

We have recently been investigating the binary Json format called [Smile](http://wiki.fasterxml.com/SmileFormat) with Jackson supports with just one line of code change. Based on our Guest Context dataset we found that on a record by record basis Smile was on average 30% faster at reading and writing and 50% smaller than the text Json equivalent. It is highly likely that we will use this in various sensitive locations in the future. 

Json as we have said is bad for the environment when processing it record by record transactionally. However, it is even worse when you want to perform analytics on it. When you store Json files on s3 they are stored in what’s called a "row format" (similar to if you stored Avro, Thrift, CSV etc). This means that to perform a query on the data, you must read every record (row) and filter / aggregate on it as per the query. Apache Spark has really powerful support for working with Json including schema inference for efficient internal representation and querying. However if you have billions of data points over terabytes of data you want to avoid having to read everything just to access a subset of the data which the query is looking for. This is where columnar data formats come into play. The two most well known are [Parquet](https://parquet.apache.org/) and [ORC](https://orc.apache.org/). Both store the data in columnar format instead of row format which allows for efficient querying including predicate push down and projection (column) pruning. We won’t go into the details of how this works and instead point you to one of the many excellent resources on it [here](http://www.slideshare.net/cloudera/hadoop-summit-36479635). Spark has excellent support for Parquet and what is even nicer is that you can read your data in json, have it infer the schema and write your data in Parquet for optimal analysis. Of course this comes with its own issues and should be used wisely. We will discuss this topic in more detail in a later blog post. 

### Querying the Data Lake

Having all the data in the data lake is great but you still need to be able to query it. For best performance you should have your data stored as parquet, orc or some other columnar storage format. It will still work with Json or avro but with row formats it will be need to read every record fully to execute a query on that data (as previous discussed). Thankfully there are a range of tools now which make querying directly against raw parquet files stored on s3 both simple and provide extremely good performance. 

Spark allows you to do this via the Spark shell but it not very user friendly (nor is it meant to be). That’s where we employed [Apache Zeppelin](https://zeppelin.apache.org/) which is essentially an Exploratory Data Analysis (EDA) tool similar to Jupyter notebooks and integrates very tightly with Spark. You can query using any of Sparks supported languages including Python, Scala and SQL. It also connects to other tools and frameworks like Presto, Flink, etc.

However, when it comes to providing SQL access to your data however you should choose an ANSI SQL compatible engine. There are plenty of ANSI SQL compatible engines available now such as [Presto](https://prestodb.io/), [Drill](https://drill.apache.org/) and [Impala](https://impala.incubator.apache.org/) to name but a few. These tools are also designed to work directly against your data sources such as a data lake (i.e. query the data where it is) and work efficiently with nested data structures such as arrays of structs etc. This all means less ETL work and your data models can match what you accept at your APIs! There is no longer any need to remodel or denormalise your data to provide fast and complex analytics against it. We chose to work with Presto. Presto is designed for interactive queries in the millisecond range on Petabytes of data. It supports the concept of querying your data in place and connecting to multiple datastores. AWS have announced Athena which is a hosted Presto solution. It is currently not available outside of the us but when it becomes available it will enable companies to take advantage of Presto and query their Data Lakes without having to manage a Presto setup themselves. Note that EMR 5.x provides presto as an out-of-the-box application also. We will detail our experience running Presto on EMR against Parquet in a follow up blog post to this series. 

