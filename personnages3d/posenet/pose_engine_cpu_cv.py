# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
from tflite_runtime.interpreter import Interpreter
import time

KEYPOINTS = (
  'nose',
  'left eye',
  'right eye',
  'left ear',
  'right ear',
  'left shoulder',
  'right shoulder',
  'left elbow',
  'right elbow',
  'left wrist',
  'right wrist',
  'left hip',
  'right hip',
  'left knee',
  'right knee',
  'left ankle',
  'right ankle'
)

KEYPOINTS_DICT = dict(zip(KEYPOINTS, range(len(KEYPOINTS))))

poseChain = (
  ('nose', 'left eye'),
  ('left eye', 'left ear'),
  ('nose', 'right eye'),
  ('right eye', 'right ear'),
  ('nose', 'left shoulder'),
  ('left shoulder', 'left elbow'),
  ('left elbow', 'left wrist'),
  ('left shoulder', 'left hip'),
  ('left hip', 'left knee'),
  ('left knee', 'left ankle'),
  ('nose', 'right shoulder'),
  ('right shoulder', 'right elbow'),
  ('right elbow', 'right wrist'),
  ('right shoulder', 'right hip'),
  ('right hip', 'right knee'),
  ('right knee', 'right ankle')
)

parentToChildEdges = [KEYPOINTS_DICT[poseChain[i][1]] for i in range(len(poseChain))]
childToParentEdges = [KEYPOINTS_DICT[poseChain[i][0]] for i in range(len(poseChain))]

class Keypoint:
    __slots__ = ['k', 'yx', 'score']

    def __init__(self, k, yx, score=None):
        self.k = k
        self.yx = yx
        self.score = score

    def __repr__(self):
        return 'Keypoint(<{}>, {}, {})'.format(self.k, self.yx, self.score)


class Pose:
    __slots__ = ['keypoints', 'score']

    def __init__(self, keypoints, score=None):
        assert len(keypoints) == len(KEYPOINTS)
        self.keypoints = keypoints
        self.score = score

    def __repr__(self):
        return 'Pose({}, {})'.format(self.keypoints, self.score)

class PoseEngine:
    """Engine used for pose tasks."""

    def __init__(self, model_path, mirror=False,
                 offsetRefineStep = 2, scoreThreshold = 0.8,
                 maxPoseDetections = 5, nmsRadius = 30, minPoseConfidence=0.15):
        """Creates a PoseEngine with given model.

        Args:
          model_path: String, path to TF-Lite Flatbuffer file.
          mirror: Flip keypoints horizontally

        Raises:
          ValueError: An error occurred when model output is invalid.
        """
        self.interpreter = Interpreter(model_path)
        self.interpreter.allocate_tensors()

        self._mirror = mirror

        self._input_tensor_shape = self.get_input_tensor_shape()
        if (self._input_tensor_shape.size != 4 or
                self._input_tensor_shape[3] != 3 or
                self._input_tensor_shape[0] != 1):
            raise ValueError(
                ('Image model should have input shape [1, height, width, 3]!'
                 ' This model has {}.'.format(self._input_tensor_shape)))
        _, self.image_height, self.image_width, self.image_depth = self.get_input_tensor_shape()

        self.heatmaps_nx = self.interpreter.get_output_details()[0]['shape'][2]
        self.heatmaps_ny = self.interpreter.get_output_details()[0]['shape'][1]
        self.heatmaps_stride_x = self.getStride(self.image_width, self.heatmaps_nx)
        self.heatmaps_stride_y = self.getStride(self.image_height, self.heatmaps_ny)
        self.quant_heatmaps_r, self.quant_heatmaps_off = self.interpreter.get_output_details()[0]['quantization']
        self.quant_offsets_short_r, self.quant_offsets_short_off = self.interpreter.get_output_details()[1]['quantization']
        self.quant_offsets_mid_r, self.quant_offsets_mid_off = self.interpreter.get_output_details()[2]['quantization']

        self.offsetRefineStep = offsetRefineStep
        self.scoreThreshold = scoreThreshold
        self.maxPoseDetections = maxPoseDetections
        self.nmsRadius = nmsRadius
        self.sqRadius = self.nmsRadius*self.nmsRadius
        self.minPoseConfidence = minPoseConfidence

        # The API returns all the output tensors flattened and concatenated. We
        # have to figure out the boundaries from the tensor shapes & sizes.
        offset = 0
        self._output_offsets = [0]
        for size in self.get_all_output_tensors_sizes():
            offset += size
            self._output_offsets.append(offset)

    def getStride(self, l, n):
        strides = (8, 16, 32)
        return strides[np.argmin(np.abs(strides - l/n))]

    def get_input_tensor_shape(self):
        return self.interpreter.get_input_details()[0]['shape']

    def get_all_output_tensors_sizes(self):
        sizes = np.array([], dtype='int32')
        for d in self.interpreter.get_output_details():
            s = np.squeeze(self.interpreter.get_tensor(d['index'])).flatten().size
            sizes = np.append(sizes, int(s))
        return sizes

    def DetectPosesInImage(self, img):
        """Detects poses in a given image.

           For ideal results make sure the image fed to this function is close to the
           expected input size - it is the caller's responsibility to resize the
           image accordingly.

        Args:
          img: numpy array containing image
        """

        # Extend or crop the input to match the input shape of the network.
        if img.shape[0] < self.image_height or img.shape[1] < self.image_width:
            img = np.pad(img, [[0, max(0, self.image_height - img.shape[0])],
                               [0, max(0, self.image_width - img.shape[1])], [0, 0]],
                         mode='constant')
        img = img[0:self.image_height, 0:self.image_width]
        assert (img.shape == tuple(self._input_tensor_shape[1:]))

        # Run the inference (API expects the data to be flattened)
        return self.ParseOutput(self.run_inference(img))

    def run_inference(self, img):
        if img.shape[0] < self.image_height or img.shape[1] < self.image_width:
            img = np.pad(img, [[0, max(0, self.image_height - img.shape[0])],
                               [0, max(0, self.image_width - img.shape[1])], [0, 0]],
                         mode='constant')
        img = img[0:self.image_height, 0:self.image_width]
        assert (img.shape == tuple(self._input_tensor_shape[1:]))

        tensor_index = self.interpreter.get_input_details()[0]['index']
        input_tensor = self.interpreter.tensor(tensor_index)
        input_tensor()[:,:,:,:] = img
        start_time = time.monotonic()
        self.interpreter.invoke()
        elapsed_ms = (time.monotonic() - start_time) * 1000
        out = np.empty(0)
        for d in self.interpreter.get_output_details():
            o = np.squeeze(self.interpreter.get_tensor(d['index'])).flatten()
            out = np.append(out, o)
        return (elapsed_ms, out)

    def logistic(self, x):
        return 1/(1+np.exp(-x))

    def isPeak(self, heatmaps_flat, index):
        maxindex = index // len(KEYPOINTS)
        maxkeypoint = index % len(KEYPOINTS)

        y_index = maxindex // self.heatmaps_nx
        x_index = maxindex % self.heatmaps_nx

        y_index_min = np.max((y_index-1, 0))
        y_index_max = np.min((y_index+1, self.heatmaps_ny-1))
        x_index_min = np.max((x_index-1, 0))
        x_index_max = np.min((x_index+1, self.heatmaps_nx-1))

        for y_current in range(y_index_min, y_index_max+1):
            for x_current in range(x_index_min, x_index_max+1):
                index_current = len(KEYPOINTS)*(y_current * self.heatmaps_nx + x_current) + maxkeypoint
                if (heatmaps_flat[index_current] > heatmaps_flat[index]) and (index_current != index):
                    return False
        return True

    def ParseOutput(self, output):
        inference_time, output = output
        outputs = [output[int(i):int(j)] for i, j in zip(self._output_offsets, self._output_offsets[1:])]

        heatmaps = outputs[0].reshape(-1, len(KEYPOINTS))
        offsets_short_y = outputs[1].reshape(-1, 2*len(KEYPOINTS))[:,0:len(KEYPOINTS)]
        offsets_short_x = outputs[1].reshape(-1, 2*len(KEYPOINTS))[:,len(KEYPOINTS):2*len(KEYPOINTS)]
        offsets_mid_fwd_y = outputs[2].reshape(-1, 4*len(poseChain))[:,0:len(poseChain)]
        offsets_mid_fwd_x = outputs[2].reshape(-1, 4*len(poseChain))[:,len(poseChain):2*len(poseChain)]
        offsets_mid_bwd_y = outputs[2].reshape(-1, 4*len(poseChain))[:,2*len(poseChain):3*len(poseChain)]
        offsets_mid_bwd_x = outputs[2].reshape(-1, 4*len(poseChain))[:,3*len(poseChain):4*len(poseChain)]
        heatmaps =  self.logistic((heatmaps - self.quant_heatmaps_off)*self.quant_heatmaps_r)
        heatmaps_flat = heatmaps.flatten()
        offsets_short_y =  (offsets_short_y - self.quant_offsets_short_off)*self.quant_offsets_short_r
        offsets_short_x =  (offsets_short_x - self.quant_offsets_short_off)*self.quant_offsets_short_r
        offsets_mid_fwd_y = (offsets_mid_fwd_y - self.quant_offsets_mid_off)*self.quant_offsets_mid_r
        offsets_mid_fwd_x = (offsets_mid_fwd_x - self.quant_offsets_mid_off)*self.quant_offsets_mid_r
        offsets_mid_bwd_y = (offsets_mid_bwd_y - self.quant_offsets_mid_off)*self.quant_offsets_mid_r
        offsets_mid_bwd_x = (offsets_mid_bwd_x - self.quant_offsets_mid_off)*self.quant_offsets_mid_r

        # Obtaining the peaks of heatmaps larger than scoreThreshold
        orderedindices = np.argsort(heatmaps_flat)[::-1]
        largeheatmaps_indices = np.empty(0, dtype='int32')
        for i in range(len(orderedindices)):
            if heatmaps_flat[orderedindices[i]] < self.scoreThreshold:
                break
            if self.isPeak(heatmaps_flat, orderedindices[i]):
                largeheatmaps_indices = np.append(largeheatmaps_indices, orderedindices[i])

        pose_list = np.full(self.maxPoseDetections*2*len(KEYPOINTS), 0.0, dtype='float32').reshape(-1, len(KEYPOINTS), 2)
        maxindex_list = np.full(self.maxPoseDetections*len(KEYPOINTS), -1, dtype='int32').reshape(-1, len(KEYPOINTS))
        score_list = np.full(self.maxPoseDetections*len(KEYPOINTS), 0.0, dtype='float32').reshape(-1, len(KEYPOINTS))
        pose_score_list = np.full(self.maxPoseDetections, 0.0, dtype='float32')

        nPoses = 0
        # obtaining at most maxPoseDetections poses
        for point in range(len(largeheatmaps_indices)):
            if nPoses >= self.maxPoseDetections:
                break

            # obtain a root canidate
            maxindex = largeheatmaps_indices[point] // len(KEYPOINTS)
            maxkeypoint = largeheatmaps_indices[point] % len(KEYPOINTS)
            y = self.heatmaps_stride_y * (maxindex // self.heatmaps_nx)
            x = self.heatmaps_stride_x * (maxindex % self.heatmaps_nx)
            y += offsets_short_y[maxindex, maxkeypoint]
            x += offsets_short_x[maxindex, maxkeypoint]

            # skip keypoint with (x, y) that is close to the existing keypoints
            skip = 0
            for p in range(nPoses):
                y_exist = pose_list[p, maxkeypoint, 0]
                x_exist = pose_list[p, maxkeypoint, 1]
                if (y_exist - y)*(y_exist - y) + (x_exist - x)*(x_exist - x) < self.sqRadius:
                    skip = 1
                    break
            if skip == 1:
                continue

            # setting the maxkeypoint as root
            pose_list[nPoses, maxkeypoint, 0] = y
            pose_list[nPoses, maxkeypoint, 1] = x
            maxindex_list[nPoses, maxkeypoint] = maxindex
            score_list[nPoses, maxkeypoint] = heatmaps[maxindex, maxkeypoint]

            # backward decoding
            for edge in reversed(range(len(poseChain))):
                sourceKeypointId = parentToChildEdges[edge]
                targetKeypointId = childToParentEdges[edge]
                if maxindex_list[nPoses, sourceKeypointId] != -1 and maxindex_list[nPoses, targetKeypointId] == -1:
                    maxindex = maxindex_list[nPoses, sourceKeypointId]
                    y = pose_list[nPoses, sourceKeypointId, 0]
                    x = pose_list[nPoses, sourceKeypointId, 1]
                    y += offsets_mid_bwd_y[maxindex, edge]
                    x += offsets_mid_bwd_x[maxindex, edge]

                    y_index = np.clip(round(y / self.heatmaps_stride_y), 0, self.heatmaps_ny-1)
                    x_index = np.clip(round(x / self.heatmaps_stride_x), 0, self.heatmaps_nx-1)
                    maxindex_list[nPoses, targetKeypointId] = self.heatmaps_nx*y_index + x_index
                    for i in range(self.offsetRefineStep):
                        y_index = np.clip(round(y / self.heatmaps_stride_y), 0, self.heatmaps_ny-1)
                        x_index = np.clip(round(x / self.heatmaps_stride_x), 0, self.heatmaps_nx-1)
                        maxindex_list[nPoses, targetKeypointId] = self.heatmaps_nx*y_index + x_index
                        y = self.heatmaps_stride_y * y_index
                        x = self.heatmaps_stride_x * x_index
                        y += offsets_short_y[maxindex_list[nPoses, targetKeypointId], targetKeypointId]
                        x += offsets_short_x[maxindex_list[nPoses, targetKeypointId], targetKeypointId]

                    pose_list[nPoses, targetKeypointId, 0] = y
                    pose_list[nPoses, targetKeypointId, 1] = x
                    score_list[nPoses, targetKeypointId] = heatmaps[maxindex_list[nPoses, targetKeypointId], targetKeypointId]

            # forward decoding
            for edge in range(len(poseChain)):
                sourceKeypointId = childToParentEdges[edge]
                targetKeypointId = parentToChildEdges[edge]
                if maxindex_list[nPoses, sourceKeypointId] != -1 and maxindex_list[nPoses, targetKeypointId] == -1:
                    maxindex = maxindex_list[nPoses, sourceKeypointId]
                    y = pose_list[nPoses, sourceKeypointId, 0]
                    x = pose_list[nPoses, sourceKeypointId, 1]
                    y += offsets_mid_fwd_y[maxindex, edge]
                    x += offsets_mid_fwd_x[maxindex, edge]

                    y_index = np.clip(round(y / self.heatmaps_stride_y), 0, self.heatmaps_ny-1)
                    x_index = np.clip(round(x / self.heatmaps_stride_x), 0, self.heatmaps_nx-1)
                    maxindex_list[nPoses, targetKeypointId] = self.heatmaps_nx*y_index + x_index
                    for i in range(self.offsetRefineStep):
                        y_index = np.clip(round(y / self.heatmaps_stride_y), 0, self.heatmaps_ny-1)
                        x_index = np.clip(round(x / self.heatmaps_stride_x), 0, self.heatmaps_nx-1)
                        maxindex_list[nPoses, targetKeypointId] = self.heatmaps_nx*y_index + x_index
                        y = self.heatmaps_stride_y * y_index
                        x = self.heatmaps_stride_x * x_index
                        y += offsets_short_y[maxindex_list[nPoses, targetKeypointId], targetKeypointId]
                        x += offsets_short_x[maxindex_list[nPoses, targetKeypointId], targetKeypointId]

                    pose_list[nPoses, targetKeypointId, 0] = y
                    pose_list[nPoses, targetKeypointId, 1] = x
                    score_list[nPoses, targetKeypointId] = heatmaps[maxindex_list[nPoses, targetKeypointId], targetKeypointId]

            # calclate pose score
            score = 0
            for k in range(len(KEYPOINTS)):
                y = pose_list[nPoses, k, 0]
                x = pose_list[nPoses, k, 1]
                closekeypoint_exists = False
                for p in range(nPoses):
                    y_exist = pose_list[p, k, 0]
                    x_exist = pose_list[p, k, 1]
                    if (y_exist - y)*(y_exist - y) + (x_exist - x)*(x_exist - x) < self.sqRadius:
                        closekeypoint_exists = True
                        break
                if not closekeypoint_exists:
                    score += score_list[nPoses, k]
            score /= len(KEYPOINTS)

            if score > self.minPoseConfidence:
                pose_score_list[nPoses] = score
                nPoses += 1
            else:
                for k in range(len(KEYPOINTS)):
                    maxindex_list[nPoses, k] = -1

        # Convert the poses to a friendlier format of keypoints with associated
        # scores.
        poses = []
        for pose_i in range(nPoses):
            keypoint_dict = {}
            for point_i, point in enumerate(pose_list[pose_i]):
                keypoint = Keypoint(KEYPOINTS[point_i], point,
                                    score_list[pose_i, point_i])
                if self._mirror: keypoint.yx[1] = self.image_width - keypoint.yx[1]
                keypoint_dict[KEYPOINTS[point_i]] = keypoint
            poses.append(Pose(keypoint_dict, pose_score_list[pose_i]))

        return poses, inference_time
