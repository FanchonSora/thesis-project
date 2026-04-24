# Nền Tảng Phân Tích MRI Não Ung Bướu

Một khuôn khổ toàn diện cho phân tích MRI não đa phương thức, bao gồm phân đoạn tự động ung bướu, tổng hợp phương thức thiếu bằng mô hình khuếch tán có điều kiện, và trực quan hóa 3D tương tác.

## Tổng Quan

Dự án này triển khai một pipeline end-to-end cho phân tích ung bướu não từ các scan MRI đa phương thức (T1, T1CE, T2, FLAIR). Hệ thống có thể xử lý dữ liệu không đầy đủ bằng cách tổng hợp các phương thức thiếu và cung cấp cả giao diện dòng lệnh và web cho phân tích và trực quan hóa.

### Tính Năng Chính

- **Phân Đoạn Đa Phương Thức**: Phân đoạn ung bướu não dựa trên 3D UNet thành các vùng có liên quan về mặt lâm sàng (ET, TC, WT)
- **Tổng Hợp Phương Thức**: Mô hình khuếch tán có điều kiện để tái tạo các chuỗi MRI thiếu
- **Pipeline Tiền Xử Lý**: Chuẩn hóa cường độ thích ứng và huấn luyện dựa trên patch
- **Giao Diện Web**: Tải lên tương tác, phân tích, và trực quan hóa 3D
- **REST API**: Truy cập lập trình để tích hợp
- **Đánh Giá Chéo Tập Dữ Liệu**: Huấn luyện trên BraTS 2021, đánh giá trên BraTS 2023

## Yêu Cầu Hệ Thống

- **HĐH**: Windows 10/11, Linux, hoặc macOS
- **Python**: 3.8-3.11
- **GPU**: NVIDIA GPU với CUDA 11.0+ (khuyến nghị cho huấn luyện/tổng hợp)
- **RAM**: 16GB+ (32GB khuyến nghị)
- **Lưu Trữ**: 50GB+ cho tập dữ liệu và mô hình

## Cài Đặt

### 1. Clone Repository

```bash
git clone https://github.com/your-repo/brain-tumor-analysis.git
cd brain-tumor-analysis
```

### 2. Tạo Môi Trường Ảo

```bash
python -m venv .venv
# Trên Windows
.venv\Scripts\activate
# Trên Linux/macOS
source .venv/bin/activate
```

### 3. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

Để hỗ trợ GPU, cài đặt PyTorch với CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Tải Xuống Mô Hình

Tải xuống các mô hình đã được huấn luyện và đặt vào các thư mục phù hợp:

- Mô hình phân đoạn: `models/segmentation_module/model-weight/final_model_unet.pth`
- Mô hình tổng hợp: `models/synthesis_module/models/` (4 file mô hình)

## Cấu Trúc Dự Án

```
├── configs/                 # File cấu hình
│   ├── pipeline_config.yaml
│   └── synthesis-models/    # Cấu hình mô hình tổng hợp
├── models/                  # Mô hình đã huấn luyện
│   ├── segmentation_module/
│   └── synthesis_module/
├── src/                     # Mã nguồn
│   ├── run_pipeline.py      # Script pipeline chính
│   ├── web_api.py          # Máy chủ API web
│   ├── preprocessing.py    # Tiền xử lý dữ liệu
│   ├── models/
│   │   └── unet3d.py       # Mô hình phân đoạn
│   ├── visualize/
│   └── web_data/           # File giao diện web
├── results/                 # Thư mục đầu ra
└── README.md               # File này
```

## Cách Sử Dung

### Giao Diện Dòng Lệnh

#### Phân Đoạn Cơ Bản

```bash
python src/run_pipeline.py \
  --case-id BraTS-GLI-00001-000 \
  --input-dir /path/to/brats/data \
  --out-dir ./results
```

#### Với Tổng Hợp (Thiếu Phương Thức)

```bash
python src/run_pipeline.py \
  --case-id BraTS-GLI-00001-000 \
  --input-dir /path/to/brats/data \
  --out-dir ./results \
  --syn-w models/synthesis_module/models
```

#### Tất Cả Tùy Chọn

```bash
python src/run_pipeline.py --help
```

### Giao Diện Web

#### Khởi Động Máy Chủ

```bash
python src/web_api.py
```

Máy chủ sẽ khởi động tại `http://localhost:8000`

#### Cách Sử Dung Web

1. Mở trình duyệt đến `http://localhost:8000`
2. Tải lên file MRI (T1, T1CE, T2, FLAIR dạng .nii hoặc .nii.gz)
3. Nhập ID trường hợp
4. Nhấn "Analyze" để bắt đầu xử lý
5. Xem kết quả:
   - Lát cắt 2D với lớp phủ phân đoạn
   - Trực quan hóa 3D não
   - Đo lường thể tích
   - Tải xuống báo cáo và mesh

### REST API

#### Khởi Động Máy Chủ API

```bash
uvicorn src.web_api:app --host 0.0.0.0 --port 8000
```

#### Các Endpoint API

- `POST /analyze`: Gửi công việc phân tích
- `GET /jobs/{job_id}/status`: Kiểm tra trạng thái công việc
- `GET /jobs/{job_id}/results`: Lấy tóm tắt kết quả
- `GET /jobs/{job_id}/file/{type}`: Tải xuống file

## Định Dạng Dữ Liệu

### Dữ Liệu Đầu Vào

- **Định Dạng**: NIfTI (.nii hoặc .nii.gz)
- **Phương Thức**: T1, T1CE, T2, FLAIR
- **Đặt Tên**: Quy ước BraTS chuẩn (vd: `BraTS-GLI-00001-000-t1.nii.gz`)
- **Độ Phân Giải**: 1mm³ đẳng hướng (tự động lấy mẫu lại nếu cần)

### Cấu Trúc Thư Mục

```
input_directory/
├── BraTS-GLI-00001-000/
│   ├── BraTS-GLI-00001-000-t1.nii.gz
│   ├── BraTS-GLI-00001-000-t1ce.nii.gz
│   ├── BraTS-GLI-00001-000-t2.nii.gz
│   └── BraTS-GLI-00001-000-flair.nii.gz
└── ...
```

## Các Module

### Module Phân Đoạn

- **Kiến Trúc**: 3D UNet với khối SE dư và cổng chú ý
- **Đầu Vào**: MRI đa phương thức 4 kênh (patch 64×64×64)
- **Đầu Ra**: Phân đoạn ung bướu theo voxel (4 lớp)
- **Huấn luyện**: Dựa trên patch với lấy mẫu tập trung ung bướu

### Module Tổng Hợp

- **Kiến Trúc**: Mô hình khuếch tán có điều kiện (DDPM)
- **Mục Đích**: Tổng hợp phương thức thiếu từ các phương thức có sẵn
- **Mô Hình**: 4 mô hình riêng biệt (một cho mỗi phương thức đích)
- **Đầu Vào**: 3 phương thức → Đầu Ra: 1 phương thức được tổng hợp

### Module Trực Quan Hóa

- **Xem 2D**: Lát cắt trục, coronal, sagittal với lớp phủ
- **Xem 3D**: Mesh não tương tác với vùng ung bướu
- **Định Dạng**: Ảnh PNG, mesh OBJ, báo cáo JSON

## Huấn Luyện

### Mô Hình Phân Đoạn

```bash
# Cần dữ liệu huấn luyện BraTS 2021
python train_segmentation.py --config configs/segmentation_config.yaml
```

### Mô Hình Tổng Hợp

```bash
cd models/synthesis_module
bash scripts/train_all_modalities.sh 5000 2
```

## Đánh Giá

### Các Chỉ Số

- **Điểm Dice**: Các vùng ET, TC, WT
- **Khoảng Cách Hausdorff**: Độ chính xác biên
- **Tương Quan Thể Tích**: Ước tính kích thước

### Xác Thực Chéo Tập Dữ Liệu

- Huấn luyện: BraTS 2021 (1251 trường hợp)
- Kiểm tra: BraTS 2021 giữ lại (20%) + BraTS 2023 (219 trường hợp)

## Khắc Phục Sự Cố

### Các Vấn Đề Thường Gặp

1. **CUDA Hết Bộ Nhớ**
   - Giảm kích thước batch trong cấu hình
   - Sử dụng chế độ CPU: `--device cpu`
   - Bật checkpoint gradient

2. **Thiếu Dependencies**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

3. **Lỗi Tải Mô Hình**
   - Xác minh đường dẫn file mô hình
   - Kiểm tra tương thích phiên bản PyTorch

4. **Vấn Đề Giao Diện Web**
   - Xóa cache trình duyệt
   - Kiểm tra console cho lỗi JavaScript
   - Đảm bảo cổng 8000 khả dụng

### Mẹo Hiệu Suất

- Sử dụng GPU để suy luận nhanh hơn
- Xử lý từng trường hợp riêng lẻ để tiết kiệm bộ nhớ
- Sử dụng LOD thấp hơn cho trực quan hóa 3D trên máy chậm

## Đóng Góp

1. Fork repository
2. Tạo nhánh tính năng: `git checkout -b feature/new-feature`
3. Commit thay đổi: `git commit -am 'Add new feature'`
4. Push lên nhánh: `git push origin feature/new-feature`
5. Gửi pull request

## Giấy Phép

Dự án này được cấp phép theo Giấy phép MIT - xem file LICENSE để biết chi tiết.

## Trích Dẫn

Nếu bạn sử dụng công trình này trong nghiên cứu, vui lòng trích dẫn:

```
@thesis{your_thesis,
  title={Phân Tích Ung Bướu Não Từ MRI Đa Phương Thức Sử Dụng Học Sâu},
  author={Tên Của Bạn},
  year={2024},
  school={Trường Đại Học Của Bạn}
}
```

## Liên Hệ

Cho câu hỏi hoặc vấn đề:
- Mở issue trên GitHub
- Email: your.email@example.com