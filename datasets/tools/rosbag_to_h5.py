"""
Adapted from Event-driven Perception for Robotics https://github.com/event-driven-robotics/importRosbag
"""

from struct import unpack
from struct import error as structError
from tqdm import tqdm

import glob
import argparse
import os
import h5py
import numpy as np
from h5_packager import H5Packager

from messageTypes.common import unpack_header
from messageTypes.dvs_msgs_EventArray import (
    importTopic as import_dvs_msgs_EventArray,
)
from messageTypes.esim_msgs_OpticFlow import (
    importTopic as import_esim_msgs_OpticFlow,
)
from messageTypes.geometry_msgs_PoseStamped import (
    importTopic as import_geometry_msgs_PoseStamped,
)
from messageTypes.geometry_msgs_Transform import (
    importTopic as import_geometry_msgs_Transform,
)
from messageTypes.geometry_msgs_TransformStamped import (
    importTopic as import_geometry_msgs_TransformStamped,
)
from messageTypes.geometry_msgs_TwistStamped import (
    importTopic as import_geometry_msgs_TwistStamped,
)
from messageTypes.sensor_msgs_CameraInfo import (
    importTopic as import_sensor_msgs_CameraInfo,
)
from messageTypes.sensor_msgs_Image import (
    importTopic as import_sensor_msgs_Image,
)
from messageTypes.sensor_msgs_Imu import importTopic as import_sensor_msgs_Imu
from messageTypes.sensor_msgs_PointCloud2 import (
    importTopic as import_sensor_msgs_PointCloud2,
)
from messageTypes.tf_tfMessage import importTopic as import_tf_tfMessage


def import_topic(topic, **kwargs):
    msgs = topic["msgs"]
    topic_type = topic["type"].replace("/", "_")
    if topic_type == "dvs_msgs_EventArray":
        topic_dict = import_dvs_msgs_EventArray(msgs, **kwargs)
    elif topic_type == "esim_msgs_OpticFlow":
        topic_dict = import_esim_msgs_OpticFlow(msgs, **kwargs)
    elif topic_type == "geometry_msgs_PoseStamped":
        topic_dict = import_geometry_msgs_PoseStamped(msgs, **kwargs)
    elif topic_type == "geometry_msgs_Transform":
        topic_dict = import_geometry_msgs_Transform(msgs, **kwargs)
    elif topic_type == "geometry_msgs_TransformStamped":
        topic_dict = import_geometry_msgs_TransformStamped(msgs, **kwargs)
    elif topic_type == "geometry_msgs_TwistStamped":
        topic_dict = import_geometry_msgs_TwistStamped(msgs, **kwargs)
    elif topic_type == "sensor_msgs_CameraInfo":
        topic_dict = import_sensor_msgs_CameraInfo(msgs, **kwargs)
    elif topic_type == "sensor_msgs_Image":
        topic_dict = import_sensor_msgs_Image(msgs, **kwargs)
    elif topic_type == "sensor_msgs_Imu":
        topic_dict = import_sensor_msgs_Imu(msgs, **kwargs)
    elif topic_type == "sensor_msgs_PointCloud2":
        topic_dict = import_sensor_msgs_PointCloud2(msgs, **kwargs)
    elif topic_type == "tf_tfMessage":
        topic_dict = import_tf_tfMessage(msgs, **kwargs)
    else:
        return None
    if topic_dict:
        topic_dict["rosbagType"] = topic["type"]
    return topic_dict


def read_file(filename):
    print("Attempting to import " + filename + " as a rosbag 2.0 file.")
    with open(filename, "rb") as file:
        # File format string
        file_format = file.readline().decode("utf-8")
        print("ROSBAG file format: " + file_format)
        if file_format != "#ROSBAG V2.0\n":
            print("This file format might not be supported")
        eof = False
        conns = []
        chunks = []
        while not eof:
            # Read a record header
            try:
                header_len = unpack("=l", file.read(4))[0]
            except structError:
                if len(file.read(1)) == 0:  # Distinguish EOF from other struct errors
                    # a struct error could also occur if the data is downloaded by one os and read by another.
                    eof = True
                    continue
            # unpack the header into fields
            header_bytes = file.read(header_len)
            fields = unpack_header(header_len, header_bytes)
            # Read the record data
            data_len = unpack("=l", file.read(4))[0]
            data = file.read(data_len)
            # The op code tells us what to do with the record
            op = unpack("=b", fields["op"])[0]
            fields["op"] = op
            if op == 2:
                # It's a message
                # AFAIK these are not found unpacked in the file
                # fields['data'] = data
                # msgs.append(fields)
                pass
            elif op == 3:
                # It's a bag header - use this to do progress bar for the read
                chunk_count = unpack("=l", fields["chunk_count"])[0]
                pbar = tqdm(total=chunk_count, position=0, leave=True)
            elif op == 4:
                # It's an index - this is used to index the previous chunk
                conn = unpack("=l", fields["conn"])[0]
                count = unpack("=l", fields["count"])[0]
                for idx in range(count):
                    time, offset = unpack("=ql", data[idx * 12 : idx * 12 + 12])
                    chunks[-1]["ids"].append((conn, time, offset))
            elif op == 5:
                # It's a chunk
                fields["data"] = data
                fields["ids"] = []
                chunks.append(fields)
                pbar.update(len(chunks))
            elif op == 6:
                # It's a chunk-info - seems to be redundant
                pass
            elif op == 7:
                # It's a conn
                # interpret data as a string containing the connection header
                conn_fields = unpack_header(data_len, data)
                conn_fields.update(fields)
                conn_fields["conn"] = unpack("=l", conn_fields["conn"])[0]
                conn_fields["topic"] = conn_fields["topic"].decode("utf-8")
                conn_fields["type"] = conn_fields["type"].decode("utf-8").replace("/", "_")
                conns.append(conn_fields)
    return conns, chunks


def break_chunks_into_msgs(chunks):
    msgs = []
    for chunk in tqdm(chunks, position=0, leave=True):
        for idx in chunk["ids"]:
            ptr = idx[2]
            header_len = unpack("=l", chunk["data"][ptr : ptr + 4])[0]
            ptr += 4
            # unpack the header into fields
            header_bytes = chunk["data"][ptr : ptr + header_len]
            ptr += header_len
            fields = unpack_header(header_len, header_bytes)
            # Read the record data
            data_len = unpack("=l", chunk["data"][ptr : ptr + 4])[0]
            ptr += 4
            fields["data"] = chunk["data"][ptr : ptr + data_len]
            fields["conn"] = unpack("=l", fields["conn"])[0]
            msgs.append(fields)
    return msgs


def rekey_conns_by_topic(conn_dict):
    topics = {}
    for conn in conn_dict:
        topics[conn_dict[conn]["topic"]] = conn_dict[conn]
    return topics


def import_rosbag(filename, **kwargs):
    print("Importing file: ", filename)
    conns, chunks = read_file(filename)
    # Restructure conns as a dictionary keyed by conn number
    conn_dict = {}
    for conn in conns:
        conn_dict[conn["conn"]] = conn
        conn["msgs"] = []
    if kwargs.get("listTopics", False):
        topics = rekey_conns_by_topic(conn_dict)
        print("Topics in the file are (with types):")
        for topicKey, topic in topics.items():
            del topic["conn"]
            del topic["md5sum"]
            del topic["msgs"]
            del topic["op"]
            del topic["topic"]
            topic["message_definition"] = topic["message_definition"].decode("utf-8")
            print("    " + topicKey + " --- " + topic["type"])
        return topics
    msgs = break_chunks_into_msgs(chunks)
    for msg in msgs:
        conn_dict[msg["conn"]]["msgs"].append(msg)
    topics = rekey_conns_by_topic(conn_dict)

    imported_topics = {}
    import_topics = kwargs.get("import_topics")
    import_types = kwargs.get("import_types")
    if import_topics is not None:
        for topic_to_import in import_topics:
            for topic_in_file in topics.keys():
                if topic_in_file == topic_to_import:
                    imported_topic = import_topic(topics[topic_in_file], **kwargs)
                    if imported_topic is not None:
                        imported_topics[topic_to_import] = imported_topic
                        del topics[topic_in_file]
    elif import_types is not None:
        for type_to_import in import_types:
            type_to_import = type_to_import.replace("/", "_")
            for topic_in_file in list(topics.keys()):
                if topics[topic_in_file]["type"].replace("/", "_") == type_to_import:
                    imported_topic = import_topic(topics[topic_in_file], **kwargs)
                    if imported_topic is not None:
                        imported_topics[topic_in_file] = imported_topic
                        del topics[topic_in_file]
    else:  # import everything
        for topic_in_file in list(topics.keys()):
            imported_topic = import_topic(topics[topic_in_file], **kwargs)
            if imported_topic is not None:
                imported_topics[topic_in_file] = imported_topic
                del topics[topic_in_file]

    print()
    if imported_topics:
        print("Topics imported are:")
        for topic in imported_topics.keys():
            print(topic + " --- " + imported_topics[topic]["rosbagType"])
            # del imported_topics[topic]['rosbagType']
        print()

    if topics:
        print("Topics not imported are:")
        for topic in topics.keys():
            print(topic + " --- " + topics[topic]["type"])
        print()

    return imported_topics


def extract_rosbag(
    rosbag_path,
    output_path,
    event_topic,
    image_topic=None,
    start_time=None,
    end_time=None,
    packager=H5Packager,
):
    ep = packager(output_path)
    t0 = -1
    sensor_size = None
    if not os.path.exists(rosbag_path):
        print("{} does not exist!".format(rosbag_path))
        return

    # import rosbag
    bag = import_rosbag(rosbag_path)

    max_events = 10000000
    xs, ys, ts, ps = [], [], [], []
    num_pos, num_neg, last_ts, img_cnt = 0, 0, 0, 0

    # event topic
    print("Processing events...")
    for i in range(0, len(bag[event_topic]["ts"])):
        timestamp = bag[event_topic]["ts"][i]
        if i == 0:
            t0 = timestamp
        last_ts = timestamp

        xs.append(bag[event_topic]["x"][i])
        ys.append(bag[event_topic]["y"][i])
        ts.append(timestamp)
        ps.append(1 if bag[event_topic]["pol"][i] else 0)

        if len(xs) == max_events:
            ep.package_events(xs, ys, ts, ps)
            del xs[:]
            del ys[:]
            del ts[:]
            del ps[:]
            print(timestamp - t0)

        if bag[event_topic]["pol"][i]:
            num_pos += 1
        else:
            num_neg += 1
        last_ts = timestamp

    if sensor_size is None:
        sensor_size = [max(xs) + 1, max(ys) + 1]
        print("Sensor size inferred from events as {}".format(sensor_size))

    ep.package_events(xs, ys, ts, ps)
    del xs[:]
    del ys[:]
    del ts[:]
    del ps[:]

    # image topic
    if image_topic is not None:
        print("Processing images...")
        for i in range(0, len(bag[image_topic]["ts"])):
            timestamp = bag[image_topic]["ts"][i]
            t0 = timestamp if timestamp < t0 else t0
            last_ts = timestamp if timestamp > last_ts else last_ts
            image = bag[image_topic]["frames"][i]
            ep.package_image(image, timestamp, img_cnt)
            sensor_size = image.shape
            img_cnt += 1

    ep.add_metadata(
        num_pos,
        num_neg,
        last_ts - t0,
        t0,
        last_ts,
        img_cnt,
        sensor_size,
    )


def extract_rosbags(rosbag_paths, output_dir, event_topic, image_topic):
    for path in rosbag_paths:
        bagname = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, "{}.h5".format(bagname))
        print("Extracting {} to {}".format(path, out_path))
        extract_rosbag(path, out_path, event_topic, image_topic=image_topic)


if __name__ == "__main__":
    """
    Tool for converting rosbag events to an efficient HDF5 format that can be speedily
    accessed by python code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="ROS bag file to extract or directory containing bags")
    parser.add_argument(
        "--output_dir",
        default="/tmp/extracted_data",
        help="Folder where to extract the data",
    )
    parser.add_argument("--event_topic", default="/dvs/events", help="Event topic")
    parser.add_argument(
        "--image_topic",
        default=None,
        help="Image topic (if left empty, no images will be collected)",
    )
    args = parser.parse_args()

    print("Data will be extracted in folder: {}".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if os.path.isdir(args.path):
        rosbag_paths = sorted(glob.glob(os.path.join(args.path, "*.bag")))
    else:
        rosbag_paths = [args.path]
    extract_rosbags(rosbag_paths, args.output_dir, args.event_topic, args.image_topic)
