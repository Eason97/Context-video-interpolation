import torch
import time
import argparse
from model_backwarp import GridNet
from dataset import vimeo_dataset
from torch.utils.data import DataLoader
import os
from utils_test import predict, to_psnr,to_ssim_skimage
from test_dataloader import vimeo_test_dataset
from tensorboardX import SummaryWriter

# --- Parse hyper-parameters  train --- #
parser = argparse.ArgumentParser(description='context-aware')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-3, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=8, type=int)
parser.add_argument('-train_epoch', help='Set the training epoch', default=50, type=int)
parser.add_argument('--train_dataset', type=str, default='/home/fum16/softmax_fitst_version/vimeo_triplet/sequences/')
parser.add_argument('--out_dir', type=str, default='./output_result')
parser.add_argument('--load_model', type=str, default=None)
# --- Parse hyper-parameters  test --- #
parser.add_argument('--test_data_dir', type=str, default='/home/fum16/softmax_fitst_version/vimeo_triplet/sequences/')
parser.add_argument('--test_input', type=str, default='/home/fum16/softmax_fitst_version/vimeo_triplet/sequences/')
parser.add_argument('--test_gt', type=str, default='/home/fum16/softmax_fitst_version/vimeo_triplet/sequences/')
parser.add_argument('--predict_result', type=str, default='./output_result/picture/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)

args = parser.parse_args()
# --- train --- #
learning_rate = args.learning_rate
train_batch_size = args.train_batch_size
train_epoch= args.train_epoch
train_data_dir= args.train_dataset
# --- test --- #
test_input = args.test_input
test_gt = args.test_gt
predict_result= args.predict_result
test_data_dir= args.test_data_dir
test_batch_size=args.test_batch_size

# --- output picture and check point --- #
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
result_dir = args.out_dir + '/picture'
ckpt_dir = args.out_dir + '/checkpoint'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
logfile = open(args.out_dir + '/log.txt', 'w')
logfile.write('batch_size: ' + str(train_batch_size) + '\n')

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# --- Define the network --- #
net = GridNet(134, 3)

# --- Build optimizer --- #
optimizer = torch.optim.Adamax(net.parameters(), lr=0.001, betas=(0.9, 0.999))

# --- Calculate all trainable parameters in network --- #
pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total_params: {}".format(pytorch_total_params))

# --- Load training data --- #
dataset = vimeo_dataset(train_data_dir)
train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=True)
# --- Load testing data --- #
test_dataset = vimeo_test_dataset(test_data_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)

max_step = train_loader.__len__()
criterion=torch.nn.L1Loss()

# --- Multi-GPU --- #
gpu_nums=torch.cuda.device_count()
net = net.to(device)
net = torch.nn.DataParallel(net, device_ids=device_ids)
writer = SummaryWriter()
# --- Strat training --- #
iteration = 0


for epoch in range(train_epoch):
    start_time = time.time()
    net.train()
    for batch_idx, (frame1, frame2, frame3) in enumerate(train_loader):


        iteration +=1
        frame1 = frame1.to(device)
        frame3 = frame3.to(device)
        gt = frame2.to(device)


        # --- Zero the parameter gradients --- #
        optimizer.zero_grad()
        # --- Forward + Backward + Optimize --- #
        output = net(frame1,frame3,train_batch_size//gpu_nums)
        print(output.shape)
        loss = criterion(output, gt)
        loss.backward()
        optimizer.step()
        if iteration%100==0:
            frame_debug = torch.cat((frame1, output, gt, frame3), dim =0)
            writer.add_images('train_debug_img', frame_debug, iteration)
        writer.add_scalars('training', {'training loss':loss.item()
                                    }, iteration)
    if epoch % 1 == 0:
        print('we are testing on epoch: ' + str(epoch))
        with torch.no_grad():
            psnr_list = []
            ssim_list = []
            for batch_idx, (frame1, frame2, frame3) in enumerate(test_loader):
                frame1 = frame1.to(torch.device('cuda'))
                frame3 = frame3.to(torch.device('cuda'))
                gt = frame2.to(torch.device('cuda'))
                # print(frame1)

                frame_out = net(frame1, frame3)
                psnr_list.extend(to_psnr(frame_out, gt))
                ssim_list.extend(to_ssim_skimage(frame_out, gt))

            avr_psnr = sum(psnr_list) / len(psnr_list)
            avr_ssim = sum(ssim_list) / len(ssim_list)
                # print(frame_out)
            frame_debug = torch.cat((frame1, frame_out, gt, frame3), dim =0)
            #print(frame_debug.size())
            writer.add_images('my_image_batch', frame_debug, epoch)
            writer.add_scalars('testing', {'testing psnr':avr_psnr,
                'testing ssim': avr_ssim,
                                    }, epoch)

            torch.save(net.state_dict(), 'epoch'+str(epoch) + '.pkl')
logfile.close()
writer.close()


