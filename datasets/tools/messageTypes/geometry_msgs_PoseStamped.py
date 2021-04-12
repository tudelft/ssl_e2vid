"""
Adapted from Event-driven Perception for Robotics https://github.com/event-driven-robotics/importRosbag under GNU General Public License.
"""

from tqdm import tqdm
import numpy as np

from .common import unpackRosString, unpackRosTimestamp, unpackRosFloat64Array


def importTopic(msgs, **kwargs):
    # if 'Stamped' not in kwargs.get('messageType', 'Stamped'):
    #    return interpretMsgsAsPose6qAlt(msgs, **kwargs)
    sizeOfArray = 1024
    tsAll = np.zeros((sizeOfArray), dtype=np.float64)
    poseAll = np.zeros((sizeOfArray, 7), dtype=np.float64)
    for idx, msg in enumerate(tqdm(msgs, position=0, leave=True)):
        if sizeOfArray <= idx:
            tsAll = np.append(tsAll, np.zeros((sizeOfArray), dtype=np.float64))
            poseAll = np.concatenate((poseAll, np.zeros((sizeOfArray, 7), dtype=np.float64)))
            sizeOfArray *= 2
        # TODO: maybe implement kwargs['useRosMsgTimestamps']
        data = msg["data"]
        # seq = unpack('=L', data[0:4])[0]
        tsAll[idx], ptr = unpackRosTimestamp(data, 4)
        frame_id, ptr = unpackRosString(data, ptr)
        poseAll[idx, :], _ = unpackRosFloat64Array(data, 7, ptr)
    # Crop arrays to number of events
    numEvents = idx + 1
    tsAll = tsAll[:numEvents]
    poseAll = poseAll[:numEvents]
    point = poseAll[:, 0:3]
    rotation = poseAll[:, [6, 3, 4, 5]]  # Switch quaternion form from xyzw to wxyz
    outDict = {"ts": tsAll, "point": point, "rotation": rotation}
    return outDict
