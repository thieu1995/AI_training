#A. Hưỡng dẫn miniconda 
- Cả pip và conda đều là packages management giành cho python. Conda có thể làm được nhiều hơn pip làm: Giải thích link bên dưới
    https://stackoverflow.com/questions/20994716/what-is-the-difference-between-pip-and-conda
    
- Sự khác nhau giữa anaconda và miniconda :
    https://stackoverflow.com/questions/45421163/anaconda-vs-miniconda

    
## 1. Install 
```youtrack
# Setup Ubuntu

sudo apt update --yes
sudo apt upgrade --yes

# Get Miniconda and make it the main Python interpreter
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda 
rm ~/miniconda.sh

echo "PATH=\$PATH:\$HOME/miniconda/bin" >> .bash_profile

```
- Kiểm tra xem cài đặt được chưa?
```youtrack
    conda -h
    conda -V
    conda --version     
```

##2. Tạo môi trường ảo dùng conda 

### 2.1 Tạo môi trường ảo 

- Tạo môi trường ảo trong conda:
```youtrack
    conda create -n myenv                   (Mặc định python giống hiện tại)
    conda create -n myenv python=3.4        (Cấu hình python riêng biệt)
    conda create -n myenv scipy             (Cấu hình với package _)
    conda create -n myenv scipy=0.15.0
```

- Hoặc tạo sau đó install các package :
```youtrack
    conda install -n myenv scipy
    conda install -n myenv scipy=0.15.0    (Khuyến khích ko nên làm thế này, Nên install cùng lúc hết những cái cần)

    conda create -n myenv python=3.4 scipy=0.15.0 astroid babel
```
    
- Nếu vậy mỗi khi tạo môi trường mới lại cần install hết lại 1 đống, rất mất thời gian và phải ghi nhớ tên chúng.
=> Ta dùng default, khi nào tạo mt mới, nó tự động cài những thằng default này.
Thêm các package vào phần create_default_packages trong file: .condarc 

- Nếu ko muốn dùng default package đối với mt đặc biệt nào đấy thì dùng:
```youtrack
    conda create --no-default-packages -n myenv python
```  

* Để tạo được mt từ file: enviroment.yml
```youtrack
    conda env create -f environment.yml (nhớ là dòng đầu tiên trong file này là tên của env)
``` 

* Tạo bản clone env 
```youtrack
    conda create --name myclone --clone myenv
```
   
### 2.2 Cách sử dụng 

- Kích hoạt env bằng cách: 
```youtrack
    source activate myenv       (Lưu path name myenv vào file hệ thống)
```
 
- Xác nhận xem env đã cài đúng chưa ?
```youtrack
    conda list          (hoặc)
    conda env list 
    conda info --envs    (đồng thời kiểm tra được đang dùng env nào)
```    

- Hủy bỏ mt :
```youtrack
    source deactivate           (Xóa path name myenv khỏi file hệ thống)
```

- Xem danh sách các package trong env 
```youtrack
    conda list -n myenv         (Nếu env chưa kích hoạt)
    conda list                  (Nếu env đã kích hoạt rồi)
    conda list -n myenv scipy   (Xem thông tin package version trong env chỉ định)
```

- Xuất ra file env để có thể share cho người khác 
```youtrack
    source activate myenv                   (vào môi trường đã active để có thể export được thông tin)
    conda env export > environment.yml      (Gửi file này cho người khác, để người ta dùng 2 để tạo env từ file này)
```

- Xóa môi trường 
```youtrack
    conda remove --name myenv --all     (hoặc)
    conda env remove --name myenv   
    
    conda info --envs   (kiểm tra xem đã xóa chưa)
```

- Giờ mỗi lần chạy file python phải vào môi trường ảo của nó như sau
```youtrack
    source ~/.bashrc                # Khởi tạo conda, nếu đã khởi tạo rồi thì thôi 
    source activate ai_env 
    python ... tên file chạy   
```


#B. Hướng dẫn pip 
```youtrack

sudo apt-get install python3-pip

pip3 install virtualenv 

virtualenv venv 
virtualenv -p /usr/bin/python2.7 venv

Active your virtual environment:    source venv/bin/activate
To deactivate:                      deactivate
Create virtualenv using Python3:    virtualenv -p python3 myenv


```
