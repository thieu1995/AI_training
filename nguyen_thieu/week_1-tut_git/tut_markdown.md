# Hướng dẫn sử dụng Markdown (Các file dạng .md)

##1. Ngữ pháp cơ bản của Markdown
Tất cả cú pháp của Markdown đều rất rõ ràng và trong sáng. Nếu bạn có thể nhớ 1 lần, bạn sẽ nhớ được mãi mãi. 

###1.1 Tiêu đề
Các lớp tiêu đề h1,h2,h3 cho đến h6 có thể viết được bằng cách thêm số lượng ký tự # tương ứng vào đầu dòng. Một ký tự # tương đương với h1, 2 ký tự # tương đương với h2 ... Tuy vậy để viết một bài viết dễ đọc thì hiếm khi cần dùng đến quá 3 ký tự này.

```youtrack
# Header h1
## Header h2
### Header h3
```

### 1.2 Bôi đậm và in nghiêng
Kẹp một từ ở đầu và cuối bằng 1 ký tự * để in nghiêng, 2 ký tự ** để **bôi đậm**, và 3 ký tự *** để ***vừa in nghiêng vừa bôi đậm.***
Nếu muốn bạn có thể dùng gạch dưới _ thay cho dấu sao *. Ngoài ra chữ có thể ~~gạch ngang~~ bằng 2 dấu ~~.

```youtrack
**Bold** and *italic* and ***both***.
__Bold__ and _talic_ and ___both___

~~strike me~~
```

### 1.3 Link
Viết link trong markdown bằng cách cho alt text vào trong ngoặc vuông[] và link thật vào trong ngoặc đơn (). Ví dụ ở đây giống hệt ví dụ đầu tiên về John_Gruber ở đoạn trên.
Ngoài ra bạn có thể thêm tiêu đề cho link bằng cách thêm "title" trong mô tả bên trong ngoặc đơn.
```youtrack
[John_Gruber](https://en.wikipedia.org/wiki/John_Gruber)
[John_Gruber](https://en.wikipedia.org/wiki/John_Gruber "Markdown Creator")
```

### 1.4 Hình ảnh 
Chèn hình ảnh trong markdown chỉ khác với chèn link đôi chút. Bạn thêm ký tự ! vào đầu tiên, sau đó ghi alt text và link ảnh vào trong ngoặc vuông [] và ngoặc đơn ().

```youtrack
![Atom](https://atom.io/assets/packages-d16d6cc46fd0cf01842409577e782b74.gif)

![alt của image](your_folder/tên_image.format)
```

### 1.5 Định dạng danh sách
* Để định đạng một đoạn văn bản thành các gạch đầu dòng trong markdown, bạn dùng ký tự * và một dấu cách ở mỗi ý và dùng thêm 2 dấu cách ở đằng trước nếu muốn lùi vào một level.

```youtrack
* Ruby
* PHP
  * Laravel
  * Symfony
  * Phalcon
* Python
  * Flask
     * Jinja2
     * WSGI1.0 
  * Django 
```
sẽ trở thành:

* Ruby
* PHP
  * Laravel
  * Symfony
  * Phalcon
* Python
  * Flask
     * Jinja2
     * WSGI1.0 
  * Django 

* Nếu bạn muốn dùng số để đánh dấu thì viết số và một dấu chấm .
```youtrack
1. number one
2. number two
3. number three
```
sẽ trở thành:
1. number one
2. number two
3. number three

### 1.6 Trích dẫn
Cách viết một trích dẫn giống hệt khi bạn vẫn trả lời bình luận hay dẫn chứng trong các diễn đàn: sử dụng ký tự >

```youtrack
> Programming today is a race between software engineers striving to build bigger and better idiot-proof programs, and the Universe trying to produce bigger and better idiots. So far, the Universe is winning.
```
> Programming today is a race between software engineers striving to build bigger and better idiot-proof programs, and the Universe trying to produce bigger and better idiots. So far, the Universe is winning.


