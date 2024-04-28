import time
import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
import argparse
import json
import random
import os
from torch.utils.tensorboard import SummaryWriter
import pickle
from Model.Attention import Attention_Gated as Attention
from Model.Attention import Attention_with_Classifier
from utils import get_cam_1d
import torch.nn.functional as F
from Model.network import Classifier_1fc, DimReduction
import numpy as np
from utils import eval_metric, eval_metric_
import pandas as pd

parser = argparse.ArgumentParser(description='abc')
testMask_dir = '' ## Point to the Camelyon test set mask location

parser.add_argument('--name', default='abc', type=str)

#### IHC 
parser.add_argument('--label_col', default='label', type=str)
parser.add_argument('--isIHC', default=False, type=bool)
#####

parser.add_argument('--k_start', default=-1, type=int)
parser.add_argument('--k_end', default=-1, type=int)

parser.add_argument('--isPar', default=True, type=bool)

parser.add_argument('--splits_dir', default='', type=str)  ## Dataset_csv
parser.add_argument('--dataset_csv', default='', type=str)  ## Dataset_csv

parser.add_argument('--num_cls', default=2, type=int)
parser.add_argument('--data_dir', default='/scratch/ekansh.chauhan/FEATURES_DIRECTORY', type=str)  ## feature_dir

parser.add_argument('--in_chn', default=384, type=int)

parser.add_argument('--mDim', default=384, type=int)

parser.add_argument('--EPOCH', default=200, type=int)
parser.add_argument('--epoch_step', default='[90]', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--log_dir', default='./results', type=str)   ## log file path
parser.add_argument('--train_show_freq', default=200, type=int)
parser.add_argument('--droprate', default='0', type=float)
parser.add_argument('--droprate_2', default='0', type=float)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--batch_size_v', default=1, type=int)
parser.add_argument('--num_workers', default=4, type=int)

parser.add_argument('--numGroup', default=4, type=int)
parser.add_argument('--total_instance', default=4, type=int)
parser.add_argument('--numGroup_test', default=4, type=int)
parser.add_argument('--total_instance_test', default=4, type=int)
parser.add_argument('--grad_clipping', default=5, type=float)
parser.add_argument('--isSaveModel', action='store_false')
parser.add_argument('--debug_DATA_dir', default='', type=str)
parser.add_argument('--numLayer_Res', default=0, type=int)
parser.add_argument('--temperature', default=1, type=float)
parser.add_argument('--num_MeanInference', default=1, type=int)
parser.add_argument('--distill_type', default='AFS', type=str)   ## MaxMinS, MaxS, AFS


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(params):

    if params.isSaveModel:
        print('will save model')
    else:
        print('will not save model')

    epoch_step = json.loads(params.epoch_step)

    params.log_dir = os.path.join(params.log_dir, params.p_name)
    writer = SummaryWriter(os.path.join(params.log_dir, params.name))
    log_dir = os.path.join(params.log_dir, str(params.name))

    in_chn = params.in_chn

    classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
    attention = Attention(params.mDim).to(params.device)
    dimReduction = DimReduction(in_chn, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)
    attCls = Attention_with_Classifier(L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2).to(params.device)

    if params.num == 0:
        params.isIHC = False

    if params.isIHC:
        util_dict = {"classifier": classifier, "attention": attention, "dim_reduction": dimReduction, "att_classifier": attCls}
        if params.num != 0:
            molecular_model_path = os.path.join('/home/ekansh.chauhan/DTFD-MIL/results/IPD/three_way/normal/bt/bt_'+ str(params.num), 'best_model.pth')
        # else:
        #     molecular_model_path = "/home/ekansh.chauhan/DTFD-MIL/results/IPD/three_way/stain_net/bt/bt_0/best_model.pth"
        #     print("Loading pre-trained model from stained version")

        embed = torch.load(molecular_model_path)
        embed = transform_state_dict(embed, util_dict)

        for i in util_dict.keys():
            w_to_use = embed[i]
            util_dict[i].load_state_dict(w_to_use, strict=False)
        print("Loaded pre-trained model")


    if params.isPar:
        classifier = torch.nn.DataParallel(classifier)
        # attention = torch.nn.DataParallel(attention)
        dimReduction = torch.nn.DataParallel(dimReduction)
        # attCls = torch.nn.DataParallel(attCls)

    ce_cri = torch.nn.CrossEntropyLoss(reduction='none').to(params.device)

    if not os.path.exists(params.log_dir):
        os.makedirs(params.log_dir)


    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = os.path.join(params.log_dir, str(params.name), 'log_'+ str(params.num) + '.txt')


    save_dir = os.path.join(params.log_dir, str(params.name), 'best_model')
    z = vars(params).copy()

    with open(log_dir, 'a') as f:
        f.write(json.dumps(z))

    log_file = open(log_dir, 'a')

    SlideNames_train, Label_train = reOrganize_mDATA(params.dataset_csv, params.fold_csv, 'train', label_name=params.label_col)
    SlideNames_val, Label_val = reOrganize_mDATA(params.dataset_csv, params.fold_csv, 'val', label_name=params.label_col)
    SlideNames_test, Label_test = reOrganize_mDATA(params.dataset_csv, params.fold_csv, 'test', label_name=params.label_col)
    print(params.name, params.fold_csv)

    print_log(f'Folder name: {params.name}, Fold: {params.num + 1}', log_file)
    # print(f'Folder name: {params.name}, Fold: {params.num + 1}')
    print_log(f'training slides: {len(SlideNames_train)}, validation slides: {len(SlideNames_val)}, test slides: {len(SlideNames_test)}', log_file)
    # print(f'training slides: {len(SlideNames_train)}, validation slides: {len(SlideNames_val)}, test slides: {len(SlideNames_test)}')

    trainable_parameters = []
    trainable_parameters += list(classifier.parameters())
    trainable_parameters += list(attention.parameters())
    trainable_parameters += list(dimReduction.parameters())

    optimizer_adam0 = torch.optim.Adam(trainable_parameters, lr=params.lr,  weight_decay=params.weight_decay)
    optimizer_adam1 = torch.optim.Adam(attCls.parameters(), lr=params.lr,  weight_decay=params.weight_decay)

    is_resume = False
    if is_resume:
        dict_file = torch.load(save_dir+'_for_resume.pth')
        ### load model

    scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam0, epoch_step, gamma=params.lr_decay_ratio)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam1, epoch_step, gamma=params.lr_decay_ratio)

    early_stopping = EarlyStopping(patience = 20, stop_epoch=85)

    best_auc = 0
    best_epoch = -1
    test_auc = 0

    for ii in range(params.EPOCH):
        torch.cuda.empty_cache()
        seed_torch(seed=1)
        for param_group in optimizer_adam1.param_groups:
            curLR = param_group['lr']
            print_log(f' current learn rate {curLR}', log_file )
            # print(f' current learn rate {curLR}')
        start_time = time.time()
        train_attention_preFeature_DTFD(classifier=classifier, dimReduction=dimReduction, attention=attention, 
                                        UClassifier=attCls, mDATA_list=(SlideNames_train, Label_train),
                                        ce_cri=ce_cri, optimizer0=optimizer_adam0, optimizer1=optimizer_adam1, epoch=ii, 
                                        params=params, f_log=log_file, writer=writer, numGroup=params.numGroup, 
                                        total_instance=params.total_instance, distill=params.distill_type)
        print(f'Epoch {ii} training time: {time.time()-start_time}')
        print_log(f'>>>>>>>>>>> Validation Epoch: {ii}', log_file)
        # print(f'>>>>>>>>>>> Validation Epoch: {ii}')

        auc_val, f1_val, _, _ = test_attention_DTFD_preFeat_MultipleMean(classifier=classifier, dimReduction=dimReduction, attention=attention,
                                                           UClassifier=attCls, mDATA_list=(SlideNames_val, Label_val), 
                                                           criterion=ce_cri, epoch=ii,  params=params, f_log=log_file, writer=writer, 
                                                           numGroup=params.numGroup_test, total_instance=params.total_instance_test,
                                                           distill=params.distill_type)
        print_log(' ', log_file)

        print_log(f'>>>>>>>>>>> Test Epoch: {ii}', log_file)

        # print(' ')


        # print(f'  >>>>>>>>>>> Test Epoch: {ii}')

        tauc, tf1, preds, gts = test_attention_DTFD_preFeat_MultipleMean(classifier=classifier, dimReduction=dimReduction, attention=attention,
                                                        UClassifier=attCls, mDATA_list=(SlideNames_test, Label_test),
                                                        criterion=ce_cri, epoch=ii,  params=params, f_log=log_file, writer=writer, 
                                                        numGroup=params.numGroup_test, total_instance=params.total_instance_test, 
                                                        distill=params.distill_type)
        print_log(' ', log_file)
        # print(' ')


        tsave_dict_for_resume = {
                'epoch': ii,
                'optimizer_adam0': optimizer_adam0.state_dict(),
                'optimizer_adam1': optimizer_adam1.state_dict(),
                'classifier': classifier.state_dict(),
                'dim_reduction': dimReduction.state_dict(),
                'attention': attention.state_dict(),
                'att_classifier': attCls.state_dict()
            }
        torch.save(tsave_dict_for_resume, save_dir+'_for_resume.pth')

        if auc_val > best_auc:
            best_auc = auc_val
            best_epoch = ii
            test_auc = tauc
            test_f1 = tf1
            if params.isSaveModel:
                print('saving model')
                tsave_dict = {
                    'epoch': ii,
                    'preds': preds,
                    'gts': gts,
                    'classifier': classifier.state_dict(),
                    'dim_reduction': dimReduction.state_dict(),
                    'attention': attention.state_dict(),
                    'att_classifier': attCls.state_dict()
                }
                torch.save(tsave_dict, save_dir+'.pth')

        print_log(f' test auc: {test_auc}, test F1: {test_f1}  :  from epoch {best_epoch}', log_file)
        # print(f' test auc: {test_auc}, test F1: {test_f1}  :  from epoch {best_epoch}')

        scheduler0.step()
        scheduler1.step()

        early_stopping(ii, auc_val)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def transform_state_dict(state_dict, util_dict):
    for j in util_dict.keys():
        var = state_dict[j]
        new_state_dict = {}

        for k, v in var.items():
            name = k[7:]
            if k[:6] == "module":
                new_state_dict[name] = v
            else:
                new_state_dict[k] = v

        keys_list = list(new_state_dict.keys())

        for i in keys_list:
            if new_state_dict[i].shape[0] == 3:
                new_state_dict[i+'_not_to_use'] = new_state_dict[i]
                del new_state_dict[i]
        state_dict[j] = new_state_dict
    return state_dict


def test_attention_DTFD_preFeat_MultipleMean(mDATA_list, classifier, dimReduction, attention, UClassifier, epoch, 
                                             criterion=None,  params=None, f_log=None, writer=None, numGroup=3, 
                                             total_instance=3, distill='MaxMinS'):

    classifier.eval()
    attention.eval()
    dimReduction.eval()
    UClassifier.eval()

    sl_names = []
    SlideNames, Label = mDATA_list
    instance_per_group = total_instance // numGroup

    test_loss0 = AverageMeter()
    test_loss1 = AverageMeter()

    gPred_0 = torch.FloatTensor().to(params.device)
    gt_0 = torch.LongTensor().to(params.device)
    gPred_1 = torch.FloatTensor().to(params.device)
    gt_1 = torch.LongTensor().to(params.device)

    with torch.no_grad():

        numSlides = len(SlideNames)
        numIter = numSlides // params.batch_size_v
        tIDX = list(range(numSlides))

        for idx in range(numIter):
            
            tidx_slide = tIDX[idx * params.batch_size_v:(idx + 1) * params.batch_size_v]
            slide_names = [SlideNames[sst] for sst in tidx_slide]
            tlabel = [Label[sst] for sst in tidx_slide]
            label_tensor = torch.LongTensor(tlabel).to(params.device)
            batch_feat = [ torch.load(os.path.join(params.data_dir, 'pt_files', '{}.pt'.format(SlideNames[sst]))).to(params.device) for sst in tidx_slide ]
            sl_names.append(SlideNames[sst] for sst in tidx_slide)

            # print('batch_feat: ', batch_feat[0].shape)
            # for sst in tidx_slide:
                # print('slidenames: ', SlideNames[sst])

            for tidx, tfeat in enumerate(batch_feat):
                # print('tfeat: ', tfeat.shape)
                tslideName = slide_names[tidx]
                tslideLabel = label_tensor[tidx].unsqueeze(0)
                midFeat = dimReduction(tfeat)
                

                AA = attention(midFeat, isNorm=False).squeeze(0)  ## N

                allSlide_pred_softmax = []

                for jj in range(params.num_MeanInference):

                    feat_index = list(range(tfeat.shape[0]))
                    random.shuffle(feat_index)
                    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
                    index_chunk_list = [sst.tolist() for sst in index_chunk_list]

                    # print('index_chunk_list: ', index_chunk_list)

                    # for tindex in index_chunk_list:
                    #     print('tindex: ', len(tindex))

                    slide_d_feat = []
                    slide_sub_preds = []
                    slide_sub_labels = []

                    for tindex in index_chunk_list:
                        slide_sub_labels.append(tslideLabel)
                        idx_tensor = torch.LongTensor(tindex).to(params.device)
                        tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)

                        tAA = AA.index_select(dim=0, index=idx_tensor)
                        tAA = torch.softmax(tAA, dim=0)
                        tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                        tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

                        tPredict = classifier(tattFeat_tensor)  ### 1 x 2
                        slide_sub_preds.append(tPredict)

                        patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                        patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                        patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

                        _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)

                        if distill == 'MaxMinS':
                            topk_idx_max = sort_idx[:instance_per_group].long()
                            topk_idx_min = sort_idx[-instance_per_group:].long()
                            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                            d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                            slide_d_feat.append(d_inst_feat)
                        elif distill == 'MaxS':
                            topk_idx_max = sort_idx[:instance_per_group].long()
                            topk_idx = topk_idx_max
                            d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                            slide_d_feat.append(d_inst_feat)
                        elif distill == 'AFS':
                            slide_d_feat.append(tattFeat_tensor)

                    slide_d_feat = torch.cat(slide_d_feat, dim=0)
                    # print('slide_d_feat: ', slide_d_feat)
                    slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
                    slide_sub_labels = torch.cat(slide_sub_labels, dim=0)

                    gPred_0 = torch.cat([gPred_0, slide_sub_preds], dim=0)
                    gt_0 = torch.cat([gt_0, slide_sub_labels], dim=0)
                    loss0 = criterion(slide_sub_preds, slide_sub_labels).mean()
                    test_loss0.update(loss0.item(), numGroup)

                    gSlidePred = UClassifier(slide_d_feat)
                    # print('gSlidePred: ', gSlidePred)
                    allSlide_pred_softmax.append(torch.softmax(gSlidePred, dim=1))
                    # print('allSlide_pred_softmax after appending: ', allSlide_pred_softmax)

                allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
                # print('allSlide_pred_softmax after cat: ', allSlide_pred_softmax)
                allSlide_pred_softmax = torch.mean(allSlide_pred_softmax, dim=0).unsqueeze(0)
                # print('allSlide_pred_softmax after mean: ', allSlide_pred_softmax)
                gPred_1 = torch.cat([gPred_1, allSlide_pred_softmax], dim=0)
                # print('gPred_1: ', gPred_1)
                gt_1 = torch.cat([gt_1, tslideLabel], dim=0)
                # print('gt_1: ', gt_1)
                loss1 = F.nll_loss(allSlide_pred_softmax, tslideLabel)
                test_loss1.update(loss1.item(), 1)

    gPred_0 = torch.softmax(gPred_0, dim=1)
    # gPred_0 = gPred_0[:, -1]
    # print('1+: ', gPred_0[:10], gt_0[:10])
    # gPred_1 = gPred_1[:, -1]
    # print('2: ', gPred_1[:10], gt_1[:10])
    if params.num_cls == 2:
        macc_0, mprec_0, mrecal_0, mF1_0, auc_0 = eval_metric(gPred_0, gt_0)
        macc_1, mprec_1, mrecal_1, mF1_1, auc_1 = eval_metric(gPred_1, gt_1)
    else:
        macc_0, mprec_0, mrecal_0, mF1_0, auc_0 = eval_metric_(gPred_0, gt_0)
        macc_1, mprec_1, mrecal_1, mF1_1, auc_1 = eval_metric_(gPred_1, gt_1)

    # print(f'  First-Tier acc {macc_0}, precision {mprec_0}, recall {mrecal_0}, F1 {mF1_0}, AUC {auc_0}')
    # print(f'  Second-Tier acc {macc_1}, precision {mprec_1}, recall {mrecal_1}, F1 {mF1_1}, AUC {auc_1}')

    print_log(f'  First-Tier acc {macc_0}, precision {mprec_0}, recall {mrecal_0}, F1 {mF1_0}, AUC {auc_0}', f_log)
    print_log(f'  Second-Tier acc {macc_1}, precision {mprec_1}, recall {mrecal_1}, F1 {mF1_1}, AUC {auc_1}', f_log)

    writer.add_scalar(f'auc_0 ', auc_0, epoch)
    writer.add_scalar(f'auc_1 ', auc_1, epoch)
    writer.add_scalar(f'F1_0 ', mF1_0, epoch)
    writer.add_scalar(f'F1_1 ', mF1_1, epoch)
    writer.add_scalar(f'Acc_0 ', macc_0, epoch)
    writer.add_scalar(f'Acc_1 ', macc_1, epoch)

    return auc_1, mF1_1, (gPred_0, gPred_1), (gt_0, gt_1)



def train_attention_preFeature_DTFD(mDATA_list, classifier, dimReduction, attention, UClassifier,  optimizer0, optimizer1, 
                                    epoch, ce_cri=None, params=None, f_log=None, writer=None, numGroup=3, total_instance=3, 
                                    distill='MaxMinS'):

    SlideNames_list, Label_dict = mDATA_list

    classifier.train()
    dimReduction.train()
    attention.train()
    UClassifier.train()

    instance_per_group = total_instance // numGroup

    Train_Loss0 = AverageMeter()
    Train_Loss1 = AverageMeter()

    numSlides = len(SlideNames_list)
    numIter = numSlides // params.batch_size

    tIDX = list(range(numSlides))
    random.shuffle(tIDX)

    for idx in range(numIter):

        tidx_slide = tIDX[idx * params.batch_size:(idx + 1) * params.batch_size]

        tslide_name = [SlideNames_list[sst] for sst in tidx_slide]
        tlabel = [Label_dict[sst] for sst in tidx_slide]
        label_tensor = torch.LongTensor(tlabel).to(params.device)

        for tidx, (tslide, slide_idx) in enumerate(zip(tslide_name, tidx_slide)):
            tslideLabel = label_tensor[tidx].unsqueeze(0)

            slide_pseudo_feat = []
            slide_sub_preds = []
            slide_sub_labels = []

            tfeat_tensor = torch.load(os.path.join(params.data_dir, 'pt_files', '{}.pt'.format(tslide)))
            tfeat_tensor = tfeat_tensor.to(params.device)

            feat_index = list(range(tfeat_tensor.shape[0]))
            random.shuffle(feat_index)
            index_chunk_list = np.array_split(np.array(feat_index), numGroup)
            index_chunk_list = [sst.tolist() for sst in index_chunk_list]

            for tindex in index_chunk_list:
                slide_sub_labels.append(tslideLabel)
                subFeat_tensor = torch.index_select(tfeat_tensor, dim=0, index=torch.LongTensor(tindex).to(params.device))
                tmidFeat = dimReduction(subFeat_tensor)
                tAA = attention(tmidFeat).squeeze(0)
                tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
                tPredict = classifier(tattFeat_tensor)  ### 1 x 2
                slide_sub_preds.append(tPredict)

                patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

                _, sort_idx = torch.sort(patch_pred_softmax[:,-1], descending=True)
                topk_idx_max = sort_idx[:instance_per_group].long()
                topk_idx_min = sort_idx[-instance_per_group:].long()
                topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

                MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)   ##########################
                max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
                af_inst_feat = tattFeat_tensor

                if distill == 'MaxMinS':
                    slide_pseudo_feat.append(MaxMin_inst_feat)
                elif distill == 'MaxS':
                    slide_pseudo_feat.append(max_inst_feat)
                elif distill == 'AFS':
                    slide_pseudo_feat.append(af_inst_feat)

            slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs

            ## optimization for the first tier
            slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x fs
            slide_sub_labels = torch.cat(slide_sub_labels, dim=0) ### numGroup
            loss0 = ce_cri(slide_sub_preds, slide_sub_labels).mean()
            optimizer0.zero_grad()
            loss0.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(attention.parameters(), params.grad_clipping)
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), params.grad_clipping)
            optimizer0.step()

            ## optimization for the second tier
            gSlidePred = UClassifier(slide_pseudo_feat)
            loss1 = ce_cri(gSlidePred, tslideLabel).mean()
            optimizer1.zero_grad()
            loss1.backward()
            torch.nn.utils.clip_grad_norm_(UClassifier.parameters(), params.grad_clipping)
            optimizer1.step()

            Train_Loss0.update(loss0.item(), numGroup)
            Train_Loss1.update(loss1.item(), 1)

        if idx % params.train_show_freq == 0:
            tstr = 'epoch: {} idx: {}'.format(epoch, idx)
            tstr += f' First Loss : {Train_Loss0.avg}, Second Loss : {Train_Loss1.avg} '
            # print(tstr)
            print_log(tstr, f_log)

    writer.add_scalar(f'train_loss_0 ', Train_Loss0.avg, epoch)
    writer.add_scalar(f'train_loss_1 ', Train_Loss1.avg, epoch)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=15, stop_epoch=50):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.counter = 0
        self.best_score = 0
        self.early_stop = False

    def __call__(self, epoch, val_auc):

        score = val_auc

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_log(tstr, f):
    # with open(dir, 'a') as f:
    f.write('\n')
    f.write(tstr)
    print(tstr)


def reOrganize_mDATA_test(mDATA):

    tumorSlides = os.listdir(testMask_dir)
    tumorSlides = [sst.split('.')[0] for sst in tumorSlides]

    SlideNames = []
    FeatList = []
    Label = []
    for slide_name in mDATA.keys():
        SlideNames.append(slide_name)

        if slide_name in tumorSlides:
            label = 1
        else:
            label = 0
        Label.append(label)

        patch_data_list = mDATA[slide_name]
        featGroup = []
        for tpatch in patch_data_list:
            tfeat = torch.from_numpy(tpatch['feature'])
            featGroup.append(tfeat.unsqueeze(0))
        featGroup = torch.cat(featGroup, dim=0) ## numPatch x fs
        FeatList.append(featGroup)

    return SlideNames, FeatList, Label


def reOrganize_mDATA(dataset_csv, fold_csv, set_type, label_name='label'):

    SlideNames = []
    Label = []

    mDATA_slides = pd.read_csv(fold_csv)
    mDATA_label = pd.read_csv(dataset_csv)

    temp_SlideNames = mDATA_slides[set_type]

    mDATA = mDATA_label[mDATA_label['slide_id'].isin(temp_SlideNames)]

    mapping = {'subtype_1': 0, 'subtype_2': 1, 'subtype_3': 2}
    mDATA = mDATA.replace({label_name: mapping})
    
    SlideNames = mDATA['slide_id'].tolist()
    Label = mDATA[label_name].tolist()

    ## to test
    # print('SlideNames: ', SlideNames, 'Label: ', Label)

    return SlideNames, Label

if __name__ == "__main__":
    seed_torch(seed=1)
    params = parser.parse_args()
    params.p_name = params.name

    if params.k_start == -1:
        start = 0
    else:
        start = params.k_start
    if params.k_end == -1:
        end = 10
    else:
        end = params.k_end
    
    list_of_nums = [0,1,2,3,4,5,6,7,8,9]

    fold_lst = list_of_nums[start:end]

    if not os.path.isdir(params.log_dir):
        os.makedirs(params.log_dir)

    tmp_log_dir = params.log_dir

    for i in fold_lst:
        params.num = i
        params.log_dir = tmp_log_dir
        params.fold_csv = os.path.join(params.splits_dir, 'splits_'+ str(params.num) + '.csv')
        
        params.name = params.p_name + '_' + str(params.num)
        print(params.name, params.fold_csv)
        main(params)
