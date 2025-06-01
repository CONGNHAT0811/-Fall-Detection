import streamlit as st
from datetime import datetime
import time

# --- Cấu hình trang ---
st.set_page_config(
    page_title="Giám sát an toàn người thân",
    layout="wide", # Sử dụng layout 'wide' để có không gian rộng hơn
    initial_sidebar_state="expanded" # Đã thay đổi từ "collapsed" thành "expanded"
)

# --- Trạng thái session ---
# camera_mode: "AI" (hiển thị ảnh cảnh báo) hoặc "Live" (hiển thị video trực tiếp)
if 'camera_mode' not in st.session_state:
    st.session_state.camera_mode = "AI"

# --- Sidebar (có thể dùng để thêm cài đặt hoặc thông tin khác) ---
with st.sidebar:
    st.header("⚙️ Cài đặt & Thông tin")
    st.write("Phiên bản: 1.0.0")
    st.write("Thiết bị đang hoạt động: ESP32-CAM_001")
    st.button("Tải lại trạng thái", use_container_width=True) # Ví dụ một nút chức năng
    
    st.markdown("---") # Đường phân cách

    # --- Hướng dẫn sử dụng (Đã di chuyển vào sidebar) ---
    with st.expander("📝 Hướng dẫn sử dụng"):
        st.write("""
        Hệ thống này giúp bạn giám sát an toàn cho người thân.
        - Khi có sự kiện ngã nghi ngờ, hệ thống sẽ gửi cảnh báo tức thì.
        - Sử dụng các nút "Chế độ AI (Dự đoán)" và "Chế độ Live" để chuyển đổi giữa xem ảnh sự kiện và xem video trực tiếp.
        - Vui lòng liên hệ các số khẩn cấp nếu cần hỗ trợ.
        """)
    st.markdown("---") # Đường phân cách cuối sidebar


# --- Tiêu đề chính ---
st.markdown("<h1 style='text-align: center; color: #333;'>👵 Giám sát an toàn người thân</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Bố cục chính với cột ---
# Đổi tỷ lệ cột để phần cảnh báo và live view chiếm không gian lớn hơn
col1, col2 = st.columns([3, 1]) # Cột 1 (chứa cảnh báo & live) rộng gấp 3 lần cột 2

with col1:
    # --- Khối Chế độ xem động (thay thế phần ảnh cảnh báo và live cũ) ---
    st.markdown("<h3>🖼️ Chế độ xem Camera</h3>", unsafe_allow_html=True)

    # Các nút để chuyển đổi chế độ
    col_mode_buttons = st.columns(2)
    with col_mode_buttons[0]:
        # Nút chuyển sang chế độ AI (Dự đoán)
        if st.button("Chế độ AI (Dự đoán)", use_container_width=True, key="mode_ai"):
            st.session_state.camera_mode = "AI"
            st.rerun()
    with col_mode_buttons[1]:
        # Nút chuyển sang chế độ Live
        if st.button("Chế độ Live", use_container_width=True, key="mode_live"):
            st.session_state.camera_mode = "Live"
            st.rerun()

    st.markdown("---") # Đường phân cách dưới các nút chế độ

    # Hiển thị thông báo trạng thái trước khi hiển thị ảnh/video
    if st.session_state.camera_mode == "AI":
        st.info("Hệ thống đang ở chế độ giám sát AI.")
        st.markdown("<h4>🔔 Cảnh báo Mới Nhất</h4>", unsafe_allow_html=True)
        st.image(
            "https://placehold.co/600x400/FF5733/FFFFFF?text=CANH+BAO+NGA",
            caption=f"Hình ảnh được chụp lúc {datetime.now().strftime('%H:%M:%S')}",
            use_container_width=True
        )
    elif st.session_state.camera_mode == "Live":
        st.info("🎥 Đang phát luồng video trực tiếp từ camera...")
        st.image(
            "https://i.imgur.com/8zQxKZj.gif", # Sử dụng GIF để mô phỏng luồng trực tiếp
            caption="Luồng trực tiếp từ camera...",
            use_container_width=True
        )
        st.success("Bạn đang xem video trực tiếp. Nhấn 'Chế độ AI (Dự đoán)' để quay lại giám sát.")

    st.markdown("---") # Đường phân cách cuối cột 1

with col2:
    # --- Khối Thông tin Liên hệ Khẩn cấp ---
    st.markdown("<h3>📞 Liên hệ Khẩn cấp</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #e8f5e9; padding: 15px; border-radius: 8px; border-left: 5px solid #4CAF50; margin-bottom: 20px;">
        <p style='font-size: 16px; font-weight: bold; color: #2E7D32;'>Cần hỗ trợ ngay lập tức!</p>
        <ul style="list-style-type: none; padding: 0;">
            <li><span style='font-size: 18px;'>🚑</span> <strong>Cấp cứu:</strong> <span style='font-size: 20px; font-weight: bold; color: #D32F2F;'>115</span></li>
            <li><span style='font-size: 18px;'>👨‍👩‍👧</span> <strong>Người thân 1:</strong> <span style='font-size: 18px; font-weight: bold;'>0912 345 678</span></li>
            <li><span style='font-size: 18px;'>👩‍👩‍👧</span> <strong>Người thân 2:</strong> <span style='font-size: 18px; font-weight: bold;'>0987 654 321</span></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
        
    st.markdown("---") # Đường phân cách cuối cột

# --- Footer ---
st.caption("© 2025 Hệ thống giám sát an toàn.")
