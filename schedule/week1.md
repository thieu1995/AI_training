## GA: Week 1
#### Tìm x1 đến x50 sao cho hàm f(x) = x1^2 + x2^3 + x3^2 + x4^3 + ...+ x49^2 + x50^3 đạt Min. Với điều kiện -10 <= x1,..., x50 <= 10
######Các kỹ thuật selection:
- EBS: entropy-Boltzmann selection
- TS: Tournament Selection
- RS: Rank Selection

(Có nhưng không dùng) 
- RWS: Roulette Wheel Selection
- RdS: Random Selection

######Các kỹ thuật crossover: 
- MPC: Multi Point Crossover
- WAR: Whole Arithmetic Recombination
- OX1: Davis’ Order Crossover 

(Có nhưng không dùng) 
- UC: Uniform crossover
- OPC: One-point crossover

######Các kỹ thuật mutation:
- BFM: Bit Flip Mutation (Random Resetting: Đối với mã hóa số thực)
- SM: Swap Mutation
- ScM: Scramble Mutation
- IM: Inversion Mutation

#### Task
- hoàng quốc hảo : EBS - MPC - BFM / SM / ScM / IM 
- trần vũ đức :     EBS - WAR - BFM / SM/ ScM / IM
- nguyễn anh tú :   EBS - OX1 - BFM / SM/ ScM / IM
- hà việt tiến :    TS - MPC - BFM / SM / ScM / IM 
- bùi thoại    :    TS - WAR - BFM / SM/ ScM / IM
- Vũ đức :          TS - OX1 - BFM / SM/ ScM / IM
- Phạm Việt Cường:  RS - MPC - BFM / SM / ScM / IM 
- Trần Minh Dũng :  RS - WAR - BFM / SM/ ScM / IM
- Lành Khánh :      RS - OX1 - BFM / SM/ ScM / IM


#### Vậy là mỗi người phải code 4 model (mỗi model là tổ hợp các kĩ thuật) sau đó so sánh kết quả giữa các model với nhau 
    (Về mặt thời gian hội tụ, tốc độ hội tụ - bao nhiêu vòng lặp) Vd: Hảo 
- Model1 : EBS + MPC + BFM
- Model2 : EBS + MPC + SM
- Model3 : EBS + MPC + ScM
- Model4 : EBS + MPC + IM 



































