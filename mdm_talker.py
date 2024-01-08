# This code is based on https://github.com/openai/guided-diffusion

import glob
import os
import sys
import pdb
import os.path as osp

import os
import numpy as np
import torch
sys.path.append(os.getcwd())
import shutil
from data_loaders.humanml.data.dataset import HumanML3D

from utils.fixseed import fixseed
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion

from data_loaders.tensors import collate
from sample.generate import construct_template_variables, save_multiple_samples, load_dataset
from datetime import datetime

class MDMTalker:
    def __init__(self):
        self.args = args = generate_args()
        fixseed(args.seed)
        out_path = args.output_dir
        # args.model_path = "./mdm/humanml_trans_enc_512/model000200000.pt"
        name = os.path.basename(os.path.dirname(args.model_path))
        niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
        max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
        fps = 12.5 if args.dataset == 'kit' else 20
        self.n_frames = n_frames = min(max_frames, int(args.motion_length*fps))
        is_using_data = False
        args.text_prompt = "Running around and jump up and down"
        dist_util.setup_dist(args.device)
        if out_path == '':
            out_path = os.path.join(os.path.dirname(args.model_path),
                                    'samples_{}_{}_seed{}'.format(name, niter, args.seed))
            if args.text_prompt != '':
                out_path += '_' + args.text_prompt.replace(' ', '_').replace('.', '')
            elif args.input_text != '':
                out_path += '_' + os.path.basename(args.input_text).replace('.txt', '').replace(' ', '_').replace('.', '')
        
        # this block must be called BEFORE the dataset is loaded
        if args.text_prompt != '':
            texts = [args.text_prompt]
            args.num_samples = 1
        elif args.input_text != '':
            assert os.path.exists(args.input_text)
            with open(args.input_text, 'r') as fr:
                texts = fr.readlines()
            texts = [s.replace('\n', '') for s in texts]
            args.num_samples = len(texts)
        elif args.action_name:
            action_text = [args.action_name]
            args.num_samples = 1
        elif args.action_file != '':
            assert os.path.exists(args.action_file)
            with open(args.action_file, 'r') as fr:
                action_text = fr.readlines()
            action_text = [s.replace('\n', '') for s in action_text]
            args.num_samples = len(action_text)
            
        assert args.num_samples <= args.batch_size, \
            f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
        # So why do we need this check? In order to protect GPU from a memory overload in the following line.
        # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
        # If it doesn't, and you still want to sample more prompts, run this script with different seeds
        # (specify through the --seed flag)
        args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
        print('Loading dataset...')
        self.data = data = load_dataset(args, max_frames, n_frames)
        total_num_samples = args.num_samples * args.num_repetitions
        

        print("Creating model and diffusion...")
        self.model, self.diffusion = create_model_and_diffusion(args, data)

        print(f"Loading checkpoints from [{args.model_path}]...")
        state_dict = torch.load(args.model_path, map_location='cpu')
        load_model_wo_clip(self.model, state_dict)
        

        if args.guidance_param != 1:
            self.model = ClassifierFreeSampleModel(self.model)   # wrapping model with the classifier-free sampler
        self.model.to(dist_util.dev())
        self.model.eval()  # disable random masking

        if is_using_data:
            iterator = iter(data)
            _, self.model_kwargs = next(iterator)
        else:
            collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames}] * args.num_samples
            is_t2m = any([args.input_text, args.text_prompt])
            if is_t2m:
                # t2m
                collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
            else:
                # a2m
                action = data.dataset.action_name_to_action(action_text)
                collate_args = [dict(arg, action=one_action, action_text=one_action_text) for
                                arg, one_action, one_action_text in zip(collate_args, action, action_text)]
            _, self.model_kwargs = collate(collate_args)

    def generate_motion(self, prompts, out_path = "mdm_out", num_repetitions = 1):
        curr_date_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        
        
        args, model_kwargs, model, diffusion, data= self.args, self.model_kwargs, self.model, self.diffusion, self.data
        model_kwargs['y']['text'] = prompts
        
        fps = 12.5 if args.dataset == 'kit' else 20
        
        all_motions = []
        all_lengths = []
        all_text = []
        
        total_num_samples  = self.n_frames * num_repetitions
        batch_size = num_samples= len(prompts)

        for rep_i in range(num_repetitions):
            print(f'### Sampling [repetitions #{rep_i}]')

            # add CFG scale to batch
            if args.guidance_param != 1:
                model_kwargs['y']['scale'] = torch.ones(batch_size, device=dist_util.dev()) * args.guidance_param

            sample_fn = diffusion.p_sample_loop
            

            sample = sample_fn(
                model,
                (batch_size, model.njoints, model.nfeats, self.n_frames),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            # Recover XYZ *positions* from HumanML3D vector representation
            if model.data_rep == 'hml_vec':
                n_joints = 22 if sample.shape[1] == 263 else 21
                sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
                sample = recover_from_ric(sample, n_joints)
                sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
            rot2xyz_mask = None if rot2xyz_pose_rep == 'xyz' else model_kwargs['y']['mask'].reshape(args.batch_size, self.n_frames).bool()
            
            sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=rot2xyz_pose_rep, glob=True, translation=True,
                                jointstype='smpl', vertstrans=True, betas=None, beta=0, glob_rot=None,
                                get_rotations_back=False)

            if args.unconstrained:
                all_text += ['unconstrained'] * args.num_samples
            else:
                text_key = 'text' if 'text' in model_kwargs['y'] else 'action_text'
                all_text += model_kwargs['y'][text_key]

            all_motions.append(sample.cpu().numpy())
            all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())

            print(f"created {len(all_motions) * args.batch_size} samples")


        all_motions = np.concatenate(all_motions, axis=0)
        all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
        all_text = all_text[:total_num_samples]
        all_lengths = all_lengths * batch_size
        all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]

        ##### Convert to full SMPL
        hand_len = 0.08824
        mdm_jts = all_motions.transpose(0, 3, 1, 2).reshape(batch_size, -1, 22, 3)
        
        direction = (mdm_jts[...,  -2, :] - mdm_jts[...,  -4, :])
        left = mdm_jts[...,  -2, :] + direction/np.linalg.norm(direction) * hand_len
        direction = (mdm_jts[...,  -1, :] - mdm_jts[...,  -3, :])
        right = mdm_jts[...,  -1, :] + direction/np.linalg.norm(direction) * hand_len
        mdm_jts_smpl_24 = np.concatenate([mdm_jts, left[...,  None, :], right[..., None, :]], axis = -2)
        
        return mdm_jts_smpl_24.squeeze()


if __name__ == "__main__":
    mdm_talker = MDMTalker()
    mdm_talker.generate_motion(["Running round"])