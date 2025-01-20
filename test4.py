import cv2
import streamlit as st
from ultralytics import YOLO
import pytesseract
import time
import os
from collections import Counter, defaultdict
import re
from datetime import datetime

class RecyclingDetectorWithSessions:
    MATERIAL_CATEGORIES = {
        'PET': {'code': '1', 'value': 25},
        'HDPE': {'code': '2', 'value': 20},
        'ALU': {'code': '3', 'value': 30},
        'GLASS': {'code': '4', 'value': 15}
    }
    
    def __init__(self):
        self.yolo_model = YOLO('best.pt')
        self.setup_tesseract()
        self.required_readings = 2  # Reduced from 4
        self.min_confidence_ratio = 0.6  # Reduced from 0.75
        self.padding = 20
        
        # Initialize session state if not exists
        if 'scanned_items' not in st.session_state:
            st.session_state.scanned_items = defaultdict(list)
        if 'scanning_active' not in st.session_state:
            st.session_state.scanning_active = False
        if 'saved_images' not in st.session_state:
            st.session_state.saved_images = []
        
    def setup_tesseract(self):
        default_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            '/usr/bin/tesseract',
            '/usr/local/bin/tesseract'
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break

    def expand_bbox(self, bbox, frame_shape):
        x1, y1, x2, y2 = map(int, bbox)
        height, width = frame_shape[:2]
        x1 = max(0, x1 - self.padding)
        y1 = max(0, y1 - self.padding)
        x2 = min(width, x2 + self.padding)
        y2 = min(height, y2 + self.padding)
        return x1, y1, x2, y2

    def clean_text(self, text):
        if not text:
            return None
        cleaned = ''.join(text.split())
        cleaned = re.sub(r'[^A-Za-z0-9]', '', cleaned)
        if len(cleaned) >= 8 and any(c.isdigit() for c in cleaned):
            return cleaned
        return None

    def determine_material(self, text):
        if not text:
            return 'Unknown', 0
        material_code = str(ord(text[0]) % 4 + 1)
        material_type = next(
            (mat for mat, info in self.MATERIAL_CATEGORIES.items() 
             if info['code'] == material_code),
            'Unknown'
        )
        pfand_value = self.MATERIAL_CATEGORIES.get(material_type, {'value': 0})['value']
        return material_type, pfand_value

    def get_ocr_reading(self, frame, bbox):
        try:
            x1, y1, x2, y2 = self.expand_bbox(bbox, frame.shape)
            roi = frame[y1:y2, x1:x2]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(binary, config=custom_config).strip()
            cleaned_text = self.clean_text(text)
            
            return cleaned_text, roi
            
        except Exception as e:
            st.error(f"OCR error: {str(e)}")
            return None, None

    def generate_receipt(self):
        receipt_text = []
        receipt_text.append("**Recycling Deposit Receipt**")
        receipt_text.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        receipt_text.append("=" * 40)
        
        # Calculate totals
        material_totals = defaultdict(lambda: {'count': 0, 'value': 0})
        total_items = 0
        total_value = 0
        
        for serial, items in st.session_state.scanned_items.items():
            for item in items:
                material_type = item['material_type']
                value = item['value']
                material_totals[material_type]['count'] += 1
                material_totals[material_type]['value'] += value
                total_items += 1
                total_value += value
        
        receipt_text.append("Items by Material Type:")
        for material, data in material_totals.items():
            receipt_text.append(f"{material}: {data['count']} x {data['value']/100:.2f} €")
        
        receipt_text.append("=" * 40)
        receipt_text.append(f"Total Items: {total_items}")
        receipt_text.append(f"Total Value: {total_value/100:.2f} €")
        receipt_text.append("=" * 40)
        
        # Display receipt
        st.markdown("\n".join(receipt_text))
        
        # Display images in a compact grid
        if st.session_state.saved_images:
            st.write("Scanned Items:")
            cols = st.columns(4)
            for idx, img in enumerate(st.session_state.saved_images):
                cols[idx % 4].image(img, width=100)

    def run_detection(self, cap):
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            placeholder = st.empty()
            captured_data = []
            last_processed = time.time()
            processing_delay = 0.5
            
            while st.session_state.scanning_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera error")
                    break

                placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                    caption="Live Feed"
                )
                
                current_time = time.time()
                if current_time - last_processed >= processing_delay:
                    results = self.yolo_model.predict(frame, conf=0.5)  # Reduced confidence threshold
                    
                    if len(results[0].boxes.xyxy) > 0:
                        bbox = results[0].boxes.xyxy[0].tolist()
                        text, roi = self.get_ocr_reading(frame, bbox)
                        if text:
                            captured_data.append((text, roi))
                        last_processed = current_time
                        
                        if len(captured_data) >= self.required_readings:
                            texts = [item[0] for item in captured_data]
                            text_counts = Counter(texts)
                            most_common = text_counts.most_common(1)[0]
                            
                            if (most_common[1] / len(texts)) >= self.min_confidence_ratio:
                                final_text = most_common[0]
                                final_roi = next(item[1] for item in captured_data if item[0] == final_text)
                                
                                material_type, pfand_value = self.determine_material(final_text)
                                
                                # Add item to session state
                                st.session_state.scanned_items[final_text].append({
                                    'material_type': material_type,
                                    'value': pfand_value
                                })
                                
                                # Save the image
                                st.session_state.saved_images.append(final_roi)
                                
                                # Show success and current totals
                                placeholder.empty()
                                st.success(f"Scanned: {material_type} - {pfand_value/100:.2f} €")
                                
                                # Reset for next item
                                captured_data = []
                                
                                # Show running total
                                total = sum(item['value'] for items in st.session_state.scanned_items.values() for item in items)
                                st.write(f"Running Total: {total/100:.2f} €")
                                break
                
                time.sleep(0.03)
                
        except Exception as e:
            st.error(f"Detection error: {str(e)}")
        finally:
            if cap.isOpened():
                cap.release()

def main():
    st.title("Recycling Deposit Scanner")
    
    try:
        camera_index = st.selectbox("Select Camera", options=[0, 1, 2], index=0)
        detector = RecyclingDetectorWithSessions()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Start Scanning"):
                st.session_state.scanning_active = True
                cap = cv2.VideoCapture(camera_index)
                if not cap.isOpened():
                    st.error(f"Failed to open camera {camera_index}")
                    return
                detector.run_detection(cap)
        
        with col2:
            if st.button("Stop Scanning"):
                st.session_state.scanning_active = False
        
        with col3:
            if st.button("Print Receipt"):
                st.session_state.scanning_active = False
                detector.generate_receipt()
                
                if st.button("New Session"):
                    st.session_state.scanned_items = defaultdict(list)
                    st.session_state.saved_images = []
                    st.experimental_rerun()
            
    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()