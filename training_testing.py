from config import CONFIG
import torch
from torch import nn




# ------ TRAİN STEP FONKSİYONU YAZALIM -------

def train_step(model:torch.nn.Module,DataLoader:torch.utils.data.DataLoader,optimizer:torch.optim.Optimizer,loss_fn:torch.nn.Module,acc_fn:torch.nn.Module):
    model.train()
    train_loss,train_acc=0,0

    for batch,(x,y) in enumerate( DataLoader):
        #tahmin yap
        y_pred=model(x)
        # tahmin kayıplarını hesapla
        loss=loss_fn(y_pred,y)
        train_loss+=loss.item()
        #tahmin doğruluğunu hesapla
        train_acc=acc_fn(y_pred,y)
        # geri yayılım uygula
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%500==0:
            print(f'BATCH:{batch}, | ANLIK KAYIP:{loss.item()}, | ANLIK DOĞRULUK:{train_acc}')
    train_loss/=len(DataLoader)
    train_acc=acc_fn.compute().item()
    return train_loss,train_acc

#------------------ TEST FONKSİYONU YAZALIM --------------

def model_test_fonksiyon(model:nn.Module,data_loader:torch.utils.data.DataLoader,loss_function:torch.nn.Module,acc_function:torch.nn.Module):
    acc_function.reset()
    loss,acc=0,0
    model.eval()
    with torch.inference_mode():
        for x, y in data_loader:
            x_pred = model(x)

            loss+=loss_function(x_pred,y).item()# .item() kullanımı şart

            acc=acc_function(x_pred,y)
        # DİKKAT: Fonksiyonun içindeki parametreyi (data_loader) kullanıyoruz
        loss /= len(data_loader)
        acc =acc_function.compute().item()
    #print(f"\nModel name : {model.__class__.__name__} | Test loss: {loss:.5f}, Test acc: %{acc*100:.2f}\n")
    return loss,acc



# ------------- EĞİTİM DÖNGÜSÜ FONKSİYONU ------------
# yukarıdaki fonksiyonları bunun içerisinde kullanacağız.

def train(model:nn.Module,
          train_data_loader:torch.utils.data.DataLoader,# eğitim verisi
          test_data_loader:torch.utils.data.DataLoader,# test verisi
          loss_function:torch.nn.Module,
          acc_function:torch.nn.Module,
          optimizer:torch.optim.Optimizer,
          epochs=CONFIG.EPOCHS,
          ):

       # verileri listelere alacağız...
    result={"train_loss":[],"train_acc":[],"test_loss":[],"test_acc":[]}

    # eğitim döngüsü

    for epoch in range(epochs):
           train_loss,train_acc=train_step(model=model,
                                           DataLoader=train_data_loader,
                                           optimizer=optimizer,
                                           loss_fn=loss_function,
                                           acc_fn=acc_function)
           test_loss,test_acc=model_test_fonksiyon(model=model,
                                                   data_loader=test_data_loader,
                                                   loss_function=loss_function,
                                                   acc_function=acc_function)

           print(f'EPOCH : {epoch+1} | TRAİN_LOSS : %{train_loss:.4f} | TRAİN_ACC : %{train_acc*100:.4f} | TEST_LOSS : %{test_loss:.4f} | TEST_ACC : %{test_acc*100:.4f}')

           result["train_loss"].append(train_loss.item() if isinstance(train_loss,torch.Tensor) else train_loss)
           result["train_acc"].append(train_acc.item() if isinstance(train_acc,torch.Tensor) else train_acc)
           result["test_loss"].append(test_loss.item() if isinstance(test_loss,torch.Tensor) else test_loss)
           result["test_acc"].append(test_acc.item() if isinstance(test_acc,torch.Tensor) else test_acc)

    return result