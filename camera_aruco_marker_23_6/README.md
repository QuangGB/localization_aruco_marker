1. Install Anaconda on Ubuntu
step 1: sudo apt-get update
step 2: download anaconda từ trang chủ (https://www.anaconda.com/products/individual)
Trang hướng dẫn: https://phoenixnap.com/kb/how-to-install-anaconda-ubuntu-18-04-or-20-04

2. Chạy ROS trong môi trường ảo tạo bởi acaconda
Đối với ubuntu 18.04, các chương trình trong ROS viết bằng python 2.7, vì vậy muốn chạy môi trường ảo sử dụng python 3, lần lượt gõ lệnh:
open Terminal và gõ: export PATH=/usr/bin/anaconda/bin:$PATH
câu lệnh này giúp chạy ROS trong môi trường ảo của anaconda

3. Cài đặt opencv 4.4.0
Trước tiên phải mở môi trường ảo bằng lệnh: 'conda activate <tên môi trường>'


