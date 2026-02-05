import os.path as osp
import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np

class SOICALTASKDataLayer(data.Dataset):
    """Dataset loader for multimodal social task data."""

    def __init__(self, args, phase):
        self.dataset_name = args.dataset_name
        self.pickle_root = f'{args.root}/dataset/{self.dataset_name}/Encoder_input'
        self.phase_video_ids = getattr(args, phase + '_video_id_set')
        self.numclass = args.numclass
        self.inp_steps = args.N * 4 # features are sampled every 0.25 s
        self.inf_step = args.s * 4 # features are sampled every 0.25 s
        self.inputs = []
        self.phase = phase

        # Load annotations
        label_all_train = pickle.load(open(osp.join(self.pickle_root, f'anno_train.pickle'), 'rb'))
        label_all_test = pickle.load(open(osp.join(self.pickle_root, f'anno_test.pickle'), 'rb'))
        label_all = label_all_train.copy()
        label_all.update(label_all_test)
        self.all_video_ids = list(label_all.keys())

        # Load visual features
        feature_visual_train = pickle.load(open(osp.join(self.pickle_root, f'visual_vit_large_train.pickle'), 'rb'))
        feature_visual_test = pickle.load(open(osp.join(self.pickle_root, f'visual_vit_large_test.pickle'), 'rb'))
        self.feature_visual = feature_visual_train.copy()
        self.feature_visual.update(feature_visual_test)
        print(f'Loaded visual features!')

        # Load text features
        feature_text_train = pickle.load(open(osp.join(self.pickle_root, f'text_bert_train.pkl'), 'rb'))
        feature_text_test = pickle.load(open(osp.join(self.pickle_root, f'text_bert_test.pkl'), 'rb'))
        self.feature_text = feature_text_train.copy()
        self.feature_text.update(feature_text_test)
        for key in self.feature_text:
            for key2 in self.feature_text[key]:
                self.feature_text[key][key2] = self.feature_text[key][key2].detach().cpu()
        print(f'Loaded text features!')

        # Load audio features
        feature_audio_train = pickle.load(open(osp.join(self.pickle_root, f'audio_wav2vec2_train.pkl'), 'rb'))
        feature_audio_test = pickle.load(open(osp.join(self.pickle_root, f'audio_wav2vec2_test.pkl'), 'rb'))
        self.feature_audio = feature_audio_train.copy()
        self.feature_audio.update(feature_audio_test)
        print(f'Loaded audio features!')

        # Build dataset inputs
        for video_id in self.phase_video_ids:
            label_video = label_all[video_id]
            # Remove bg segment 
            if self.dataset_name in ["MOSEI", "MOSI"]:
                start_index, end_index = self.get_video_start_end(label_video, buffer=self.inp_steps)
            else:
                start_index, end_index = 0, len(label_video)

            start = start_index
            while start + self.inp_steps <= end_index:
                feature_step = 1 if self.phase == "train" else 4
                end = start + self.inp_steps

                label_target = label_video[end - 1]
                segment_window = self.get_segment_boundary(label_video, end, self.inf_step)

                # Downsampling MOSEI pos data to balance label
                if self.dataset_name == "MOSEI":
                    if self.phase == "train" and torch.equal(label_target, torch.tensor([1, 0, 0])) and (end - 1) % 4 != 0:
                        start += feature_step
                        continue
                # Remove bg segment
                if self.dataset_name in ["MOSI", "MOSEI"]:
                    if torch.equal(label_target, torch.tensor([0, 0, 1])):
                        start += feature_step
                        continue

                self.inputs.append([video_id, start, end, label_target, segment_window ])
                start += feature_step

    def extract_segment_feature(self, context_window, step, features):
        """Extract segment feature from candidate context window boundary."""
        start, end = context_window
        sample_index = torch.tensor(list(range(end - 1, start - 1, -step))).sort()[0]
        feature_inputs = features[sample_index]
        # Pad if needed
        if feature_inputs.shape[0] < self.N:
            feature_inputs = F.pad(feature_inputs, (0, 0, 0, self.N - feature_inputs.shape[0]), mode='constant', value=0)
        elif feature_inputs.shape[0] > self.N:
            feature_inputs = feature_inputs[:self.N]

        if feature_inputs.shape[-1] < 1024:
            feature_inputs = F.pad(feature_inputs, (0, 1024 - feature_inputs.shape[-1]),mode='constant',value=0)
        return feature_inputs

    def get_segment_boundary(self, target, end, step):
        """Retrieve current segment boundary."""
        label = target[end - 1].tolist()

        start_index = end - 1
        while start_index > 0 and target[start_index - 1].tolist() == label:
            start_index -= step
        end_index = end - 1
        while end_index < len(target) - 1 and target[end_index + 1].tolist() == label:
            end_index += step
        return start_index, end_index

    def iou(self, segment1, segment2):
        """Compute Intersection over Union (IoU) between two segments."""
        inter_start = torch.max(segment1[0], segment2[0])
        inter_end = torch.min(segment1[1], segment2[1])
        inter = (inter_end - inter_start).clamp(min=0)

        union_start = torch.min(segment1[0], segment2[0])
        union_end = torch.max(segment1[1], segment2[1])
        union = (union_end - union_start).clamp(min=1e-6)

        return inter / union

    def get_video_start_end(self, target, buffer):
        """Find the real start and end indices of non-background regions."""

        non_background = ~(target == torch.tensor([0, 0, 1])).all(dim=1)
        indices = torch.nonzero(non_background).squeeze()
        if indices.numel() == 0:
            return None, None
        
        start_index = max(indices[0].item() - buffer, 0)
        end_index = min(indices[-1].item() + buffer, target.shape[0])
        return start_index, end_index

    def __getitem__(self, index):
        (video_id, start, end, label_target, segment_window) = self.inputs[index]

        video_id_tensor = self.all_video_ids.index(video_id)
        cur_time_id_tensor = int(end // 4)
        id_tensor = torch.tensor([video_id_tensor, cur_time_id_tensor])
        segment_window = torch.tensor(segment_window)
        context_window = torch.tensor([start, end])

        visual_inputs = self.extract_segment_feature(context_window, self.inf_step, self.feature_visual[video_id]['rgb'])
        text_inputs = self.extract_segment_feature(context_window, self.inf_step, self.feature_text[video_id][2.0])
        audio_inputs = self.extract_segment_feature(context_window, self.inf_step, self.feature_audio[video_id][4.0])

        label_target = label_target[:self.numclass]
        iou_scores = self.iou(context_window, segment_window)

        return (id_tensor, visual_inputs, text_inputs, audio_inputs, label_target, iou_scores)

    def __len__(self):
        return len(self.inputs)
