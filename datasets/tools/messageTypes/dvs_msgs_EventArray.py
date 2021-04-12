"""
Adapted from Event-driven Perception for Robotics https://github.com/event-driven-robotics/importRosbag under GNU General Public License.
"""

from tqdm import tqdm
import numpy as np

# local imports

from .common import unpackRosString, unpackRosUint32


def importTopic(msgs, **kwargs):

    tsByMessage = []
    xByMessage = []
    yByMessage = []
    polByMessage = []
    for msg in tqdm(msgs, position=0, leave=True):
        # TODO: maybe implement kwargs['useRosMsgTimestamps']
        data = msg["data"]
        # seq = unpack('=L', data[0:4])[0]
        # timeS, timeNs = unpack('=LL', data[4:12])
        frame_id, ptr = unpackRosString(data, 12)
        height, ptr = unpackRosUint32(data, ptr)
        width, ptr = unpackRosUint32(data, ptr)
        numEventsInMsg, ptr = unpackRosUint32(data, ptr)
        # The format of the event is x=Uint16, y=Uint16, ts = Uint32, tns (nano seconds) = Uint32, pol=Bool
        # Unpack in batch into uint8 and then compose
        dataAsArray = np.frombuffer(data[ptr : ptr + numEventsInMsg * 13], dtype=np.uint8)
        dataAsArray = dataAsArray.reshape((-1, 13), order="C")
        # Assuming big-endian
        xByMessage.append((dataAsArray[:, 0] + dataAsArray[:, 1] * 2 ** 8).astype(np.uint16))
        yByMessage.append((dataAsArray[:, 2] + dataAsArray[:, 3] * 2 ** 8).astype(np.uint16))
        ts = (
            dataAsArray[:, 4] + dataAsArray[:, 5] * 2 ** 8 + dataAsArray[:, 6] * 2 ** 16 + dataAsArray[:, 7] * 2 ** 24
        ).astype(np.float64)
        tns = (
            dataAsArray[:, 8] + dataAsArray[:, 9] * 2 ** 8 + dataAsArray[:, 10] * 2 ** 16 + dataAsArray[:, 11] * 2 ** 24
        ).astype(np.float64)
        tsByMessage.append(ts + tns / 1000000000)  # Combine timestamp parts, result is in seconds
        polByMessage.append(dataAsArray[:, 12].astype(np.bool))
    outDict = {
        "x": np.concatenate(xByMessage),
        "y": np.concatenate(yByMessage),
        "ts": np.concatenate(tsByMessage),
        "pol": np.concatenate(polByMessage),
        "dimX": width,
        "dimY": height,
    }
    return outDict
