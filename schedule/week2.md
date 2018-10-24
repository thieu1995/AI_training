
## PSO: Week 2
- Ai chưa code xong, code xong chưa đẩy code và kết quả thì đẩy lên git.
- Chia folder rõ ràng trong git theo từng week và để riêng folder results cho từng tuần.
- Tối ưu lại code python, đang viết hàm thì chuyển về class, comment cho code.
- Phương pháp Entropy-Boltzmann Selection đang dẫn đầu trong bảng so sánh (ai muốn hiểu thêm trao đổi với Anh Tú và Trần Vũ Đức)
- Chiều thứ 3 tuần sau sẽ có 1 buổi nữa bàn về Entropy-Boltzmann.
- Nhiệm vụ mới PSO (Làm theo team, báo cáo theo team)

#### Tìm x1 đến x50 sao cho hàm f(x) = x1^2 + x2^3 + x3^2 + x4^3 + ...+ x49^2 + x50^3 đạt Min. Với điều kiện -10 <= x1,..., x50 <= 10
###### Các yếu tố quan trọng trong PSO 
- mã hóa 1 con chim, lưu lại được vị trí tốt nhất trong quá khứ, lưu lại vận tốc hiện tại của con chim
- Hàm update di chuyển của con chim dựa vào 2 yếu tố: quán tính hiện tại, ví trị tốt nhất trong qúa khứ, vị trí tốt nhất toàn dân số 
```
x(t+1) = w*v(t) + r1*c1*(pbest - x(t)) + r2*c2*(gbest - x(t))

x(t) : vị trí hiện tại 
v(t) : vận tốc hiện tại 
w: Trọng lượng của con chim hiện tại (w giảm theo epoch)
pbest: Vị trí tốt nhất trong quá khứ 
gbest: Vị trí tốt nhất của toàn dân số 
r1, r2 = 1 số uniform random thuộc (0, 1) 
c1: Tham số cognitive (nhận thức của bản thân con chim)
c2: Tham số social (ảnh hưởng của social)
```

####### Các cấu trúc social network của PSO (communication)
- Cơ bản nhất : Star  (Đọc mấy cấu trúc này trong cuốn Intelligence Computing trang 300 hoặc tìm trên mạng)
- Mở rộng: Ring, Wheel, Pyramid, Four Clusters, Von Neumann 

#### Task
- Mỗi người đều phải code mạng Star trước. Sau đó bên dưới đây mới là làm theo team.

- Lành Khánh + Trần Minh Dũng : Ring
- Vũ Anh Đức : Wheel 
- Nguyễn Anh Tú + Phạm Việt Cường: Pyramid 
- Hoàng Quốc Hảo + Hà Việt Tiến: Four Clusters 
- Trần Vũ Đức + Bùi Thoại : Von Neumann

#### Đánh giá so sánh 
- Như tuần trước, vẽ hình so sánh giữa Star, cấu trúc code theo team và thêm cả GA tuần trước đã code.
VD: Khánh phải vẽ kq so sánh giữa: GA, Star và Ring. 






































