# Breast Cancer Prediction

## Giới thiệu
Dự án này nhằm mục đích phát hiện ung thư vú bằng cách sử dụng các thuật toán học máy. Dữ liệu được sử dụng trong dự án này bao gồm các đặc trưng của các khối u vú, và mục tiêu là phân loại chúng thành lành tính hoặc ác tính.

## Cấu trúc thư mục
- **Data/**: Chứa dữ liệu thô và các tệp đã xử lý.
  - `breast_cancer.csv`: Tệp dữ liệu gốc.
  - `X_train.npy`, `X_test.npy`: Tệp chứa dữ liệu đầu vào cho mô hình.
  - `y_train.npy`, `y_test.npy`: Tệp chứa nhãn cho dữ liệu đầu vào.
  - `Data_preprocessing.ipynb`: Tệp Jupyter Notebook cho quá trình tiền xử lý dữ liệu.

- **Notebooks/**: Chứa các tệp Jupyter Notebook cho từng mô hình học máy.
  - `AdaBoost.ipynb`: Mô hình AdaBoost.
  - `Decision_tree.ipynb`: Mô hình Cây quyết định.
  - `Gradient_boosting.ipynb`: Mô hình Gradient Boosting.
  - `KNN.ipynb`: Mô hình K-Nearest Neighbors.
  - `Logistic_Regression.ipynb`: Mô hình Hồi quy logistic.
  - `OSEL.ipynb`: Mô hình OSEL.
  - `Random_Forest.ipynb`: Mô hình Rừng ngẫu nhiên.
  - `SVM.ipynb`: Mô hình Support Vector Machine.
  - `XGBoost.ipynb`: Mô hình XGBoost.

- **src/**: Chứa mã nguồn chính của dự án.
  - `models.py`: Chứa các định nghĩa mô hình.
  - `utils.py`: Chứa các hàm tiện ích.
  - `tempCodeRunnerFile.py`: Tệp tạm thời cho việc chạy mã.

## Cài đặt
1. Clone repository này về máy của bạn:
   ```bash
   git clone <repository-url>
   ```
2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```

## Sử dụng
- Mở tệp Jupyter Notebook tương ứng với mô hình bạn muốn chạy trong thư mục `Notebooks/`.
- Chạy từng ô trong notebook để thực hiện quá trình huấn luyện và đánh giá mô hình.

## Kết quả
Dự án này sẽ cung cấp các kết quả đánh giá cho từng mô hình, bao gồm độ chính xác, độ nhạy và độ đặc hiệu.

## Liên hệ
Nếu bạn có bất kỳ câu hỏi nào, vui lòng liên hệ với tôi qua email: [your-email@example.com]