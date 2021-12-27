from spatial_test import FrameWiseHGNN, get_train, get_valid
import torch
import time
import copy


def train_it(model, criterion, optimizer, scheduler, num_epochs=25, print_freq=500):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    running_loss = 0.0
    running_corrects = 0

    for i in range(len(get_train())):
        for j in range(0, len(get_train()[0]['data_X'])):
            print("reading.....")
            images_train, images_valid, lbl_train, lbl_valid = get_train()[i]['data_X'][j], get_valid()[i]['data_X'][j], get_train()[i]['lbl'],get_valid()[i]['lbl']

            net = FrameWiseHGNN()
        
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(net.parameters(),lr=args.learning_rate)
            train_loss = []
            valid_loss = []
            print_freq = 100
            num_epochs= 25
            train_acc = 0
            batch_size = 1

            # optim criterion
            for epoch in range(args.epochs):
                if epoch % print_freq == 0:
                    print('-' * 10)
                    print(f'Epoch {epoch}/{num_epochs - 1}')
                net.train()
                # forward
                train_out = net(images_train)
                loss = criterion(train_out, lbl_train)
                train_loss += loss.item() * len(lbl_train)

                train_pred = torch.argmax(train_out)
                num_acc = (train_pred == lbl_train).sum()
                train_acc += num_acc.item()

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('epoch:[{}],train loss is:{:.6f},train acc is:{:.6f}'.format(epoch,
                                                                            train_loss / (len(images_train) * batch_size),
                                                                            train_acc / (len(images_train) * batch_size)))
                
                

            # valid
            net.eval()
            valid_epoch_loss = []
            valid_out = net(images_valid)
            loss = criterion(valid_out,lbl_valid)
            valid_epoch_loss.append(loss.item())
            valid_loss.append(loss.item())
            valid_pred = torch.argmax(valid_out)