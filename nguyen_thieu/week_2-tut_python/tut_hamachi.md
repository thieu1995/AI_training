# Hướng dẫn cài đặt và truy cập vào con máy GPU sử dụng hamachi 

- Vào: https://www.vpn.net/linux -> Tải file .dev rồi click chuột phải vào để cài đặt.
 
- Cài đặt haguichi : https://help.ubuntu.com/community/Hamachi
```youtrack
    sudo add-apt-repository -y ppa:webupd8team/haguichi
    sudo apt update
    sudo apt install -y haguichi
```

- Lần đầu thì phải configure (auto). Sau đó thì click vào dấu + và tiếp là Join Network: 
```youtrack
Network Id: Team-Research-GT
Password: @hunter#123  (Cái pass này được dùng cho cả máy chủ luôn - để dễ nhớ) 
```

- Sau đó chờ anh except trên máy chủ cho chú join mạng. 

- Rồi mở terminal lên:
```youtrack
    ssh hunter@IPv4-của-máy-chủ (Cái này xem ở chỗ Haguichi - click vào hunter là thấy) 
``` 

- Lần đầu ssh sẽ hiện lỗi này: The authenticity of host 'IP ()' can't be established.
Gõ: yes 

- Lần sau thì cứ ssh bình thường là được và sẽ không cần dùng đến haguichi nữa. Haguichi thật ra nó chỉ là cái GUI trên linux để nhìn rõ được các network có sẵn.


- Link đọc thêm:
```youtrack
    http://blog.skypayjm.com/2013/03/vpn-using-hamachi-ssh-rdp-part-i.html
    https://superuser.com/questions/421074/ssh-the-authenticity-of-host-host-cant-be-established/421084
    https://help.ubuntu.com/community/Hamachi
```


