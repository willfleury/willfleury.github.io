---
layout: default
title: Speed Serving Views
description: Building Speed Serving Views for the Lambda Architecture with Cassandra Time Series
categories: [data engineering, boxever]
---

## Speed Serving Views

This post is part of a [Series]({% post_url /data-engineering/2017-03-01-overview %}) on the Lambda Architecture. It was written in collaboration with [Boxever](http://www.boxever.com) and first posted on [Medium](https://medium.com/@BoxeverTech/time-series-cassandra-speed-view-bc6581290bc1).

### Overview

As discussed in our blog on building [Batch Views]({% post_url /data-engineering/2017-03-01-batch-serving-views %}), the batch views are read-only. Therefore to enable querying and retrieving the Guest changes since the batch view was last prepared, we must implement a speed serving view. A process can then produce a real-time view of the Guest by querying the Batch and Speed serving views and merging the results. We will cover the merging process in the next blog post in this series. 

{% include image.html img="/assets/images/data-engineering/speed_views_image_0.png" title="Speed Serving View Architecture" caption="Speed Serving View Architecture" %}

As can be seen in the above diagram, the data in this view is populated directly from the changelog events we publish to Kafka (see our post on the [Changelog]({% post_url /data-engineering/2017-03-01-changelog %}) for more details). The section in blue are the components related to the speed view. As this data also makes its way into the batch serving view eventually, the data in the speed layer has a short lifetime. This means that the data in the speed view is bounded and more easily managed. 

In the above diagram, you will also notice that we have two services reading from the Kafka changelog into the dedicated speed layer database. This is to separate the different traffic models that either come from the bulk imports via the Batch APIs or from the real time updates via the Interactive and Stream APIs. This is possible as each event has provenance information that indicates the path of entry for the update. The approach allows use to manage the different traffic models more effectively and ensures the more important real time updates are not delayed when indexing them to the speed layer. The updates via the Batch API can easily be throttled if necessary.  Another solution of course is to have separate Kafka clusters and speed layer database clusters altogether for streaming and batch but this is more expensive, more complex and based on our observations of this architecture so far, unnecessary. 

### What Database did we use?

The main properties we required from a datastore for this problem were

* TTL support on inserts
* Read/Write access to a collection of change events against a single Guest key
* Highly available
* Fast (<10ms p95)

The second requirement can be solved by the majority, if not all databases. However, the TTL requirement narrows the available solutions considerably. You can try to implement the purging of the data yourself (i.e. manual TTL), but this is needless complexity that you should avoid. It is almost certainly going to be less efficient than if it supported by the database itself.

The last requirement while obvious is important to think about thoroughly. The write patterns are a continuous stream of changelog events which we write against various guest profiles. However, the read pattern is more of a time series slice where we say ‘get all of the changes for this guest in the last 24 hours’. When viewed like this the problem is really a time series problem and should be modelled as such. If the underlying database does not have good support for time series it may be difficult to make this scale and remain performant in a latency sensitive manner.

#### Key-Value Cache

There are a number of ways one can approach the problem of storing this type of timeseries data. We initially tried a naive approach of using an in memory [Redis](https://redis.io/) cache. The main decision point around it, at the time, was speed. We believed it would be the easiest solution that would guarantee query response times in the single digit millisecond range. While it satisfied the latency requirement, it only satisfied the other requirements with varying degrees of success which we will  discuss. However, the final nail in the coffin for Redis as a solution was the growth in our traffic and storage costs associated with it. When we first analysed the storage requirements to keep 72 hours worth of data in the speed view, we badly misunderestimated the growth we would see in our traffic. Within a few months, even the largest Redis instances in AWS of 64GB were not large enough to even hold 48 hours of data (Redis is a vertical scale solution - i.e. not distributed). As we had multi-zone replication enabled for these instances to support high availability, the cost became a factor and we had to rethink. We could have looked at tools like [Dynomite](http://techblog.netflix.com/2014/11/introducing-dynomite.html) from Netflix, to make Redis a horizontally scalable solution instead of vertical scale (and remove the reliance on the Redis replication model which is far from ideal). However, the costs would have remained an issue.

The Redis data structure we used was a Hash ([HSET](https://redis.io/topics/data-types)). The Hash (or HashSet) key was based on the Guest key with each entry in the HashSet being a changelog event related to that Guest. Redis Hashes support TTL via the [EXPIRE](https://redis.io/commands/expire) command. This appeared to suit our needs well. However, it soon became obvious that there were severe limitations and problems with modelling access to our changelog in this manner. The TTL for Redis Hashes is based on the parent key (the Hash itself) and not the child entries. This led to unbounded growth issues within the Hashes. If a new change event was added to a Hash, it would reset the TTL on the entire Hash. So for Guest profiles like test accounts, where they would constantly be receiving some change events, the Hash entries would never expire. In essence this is a memory leak. It required us to check the size of the Hash before adding items to it and purging the oldest manually. This was expensive with needless complexity. 

#### Time Series with Cassandra

When we re-evaluated the requirements we realised the [Cassandra](http://cassandra.apache.org/) was a better option for modelling our problem. We had concerns on the query latency at the p95 range based on the experiences with our primary Cassandra cluster. However, we also knew that when used and tuned correctly, Cassandra could perform excellently at the p95 range as we saw with its usage for our [Batch Views]({% post_url /data-engineering/2017-03-01-batch-serving-views %}). Initially, we used [m1.xlarge](https://aws.amazon.com/ec2/previous-generation/) nodes for this cluster as it was the standard node type we had for Cassandra. However, these had severe performance issues when under load. Fortunately, when we changed to the newer [m4.xlarge](https://aws.amazon.com/ec2/instance-types/) nodes with [gp2 SSDs](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EBSVolumeTypes.html) it performed brilliantly. With just 3 m4.xlarge nodes running Cassandra 2.2 we achieved p95 query response times of ~ 8ms at a sustained query rate of 500 qps and write rate of 1500 writes / second. Peaks can be multiples of this and in fact we have seen the write rates hitting 8k / second without significant degradation to query response times. So, how did we configure Cassandra to achieve this?

The data model we used is shown below. It is based on the Datastax documentation on time series modelling with Cassandra [here](https://www.datastax.com/dev/blog/datetieredcompactionstrategy). From the primary key definition, you can see that we partition the data by the guest key. Cassandra, under the hood stores the data for each key as a wide row. This ensures that all the data required to satisfy a query for a given key is located on the same partition and hence the same node. That means faster queries. The timestamp and entity ref make up the unique constraint for the primary key. We also define a clustering order based on the descending timestamp value for the record. This ensures that within each partition the records are stored in a manner such that when we read the changes for a guest, we know that they are in-order which allows for certain optimisations to be performed by the client (ignore older versions of same entity change event). It also means that when we query for changes in the last N hours, the data is already stored on disk in the most efficient way possible for reading back to the client. 

One column in our schema which is worth highlighting is the `deleted` marker. Remember we are storing our changelog as a time series in Cassandra. We never actually delete data ourselves and data is aged out by the TTL process only. Therefore, in a true changelog fashion if an entity related to a guest has been deleted, we actually write an event to the journal in Cassandra with the deleted column set. When reading the data we can then apply this delete in the client (i.e. if last journal entry for a given entity has the delete marker set then we know the entity is no longer associated with the given Guest).  

```sql
CREATE TABLE IF NOT EXISTS boxever_guest_context.journal (
  key text,
  ts timestamp,
  ref UUID,
  type text,
  event blob,
  deleted boolean,
  PRIMARY KEY ((key), ts, ref))
WITH CLUSTERING ORDER BY (ts DESC)
AND gc_grace_seconds = 0
AND default_time_to_live = 604800
AND compaction = {
'class':'DateTieredCompactionStrategy',
'timestamp_resolution':'MICROSECONDS',
'base_time_seconds':'3600',
'min_threshold' : 4,
'max_sstable_age_days':'2',
'tombstone_compaction_interval':'1'
};
```

You can also see from the schema that we have the TTL set to 7 days. This allows us to take partial time slices of the entity change log for any time period in the last 7 days. Keeping some additional headroom between the rebuild frequency of the batch serving views and the TTL in the speed layer is important so that the merging layer can dynamically expand the query window it uses based on the current active batch view dataset. Having the additional headroom makes for a very operationally stable system. We will cover this in more detail when we discuss how we merge the batch and speed views. 

#### Issues with DateTieredCompactionStrategy 

One major issue we encountered with using [DateTieredCompactionStrategy](https://issues.apache.org/jira/browse/CASSANDRA-9666) was what happens if you run a repair operation on a node. A repair operation is typically seen as a safe thing to do, however, with DateTieredCompactionStrategy, it results in an explosion of SSTables being produced which in turn causes a continuous stream of compactions to run on the node. This severely affects the responsiveness of the node as compactions put the nodes IO, CPU and memory under pressure. This in turn, affects the latency of queries, particularly badly in the higher percentiles. Some of the operational issues around DateTieredCompactionStrategy were so severe that it has been deprecated as of 3.x of Cassandra and replaced with a new [TimeWindowCompactionStrategy](https://docs.datastax.com/en/cassandra/3.0/cassandra/dml/dmlHowDataMaintain.html). We plan on leaving the 3.x series of Cassandra mature for a while before we upgrade.

### Summary

With the completion of the speed serving layer, we now have all the pieces in place to complete the lambda architecture. We simply need to merge the data in both the speed and batch serving views and provide access to it via a REST API. We will cover that next in our final blog post on this series. 

Next in the series - [Merge Serving Views]({% post_url /data-engineering/2017-03-01-merge-serving-views %})