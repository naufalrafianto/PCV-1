import cv2
from card_processor import detect_card, get_warped_card
from dataset_collector import DatasetCollector
from card_constants import print_valid_labels


def main():
    # Inisialisasi dataset collector
    collector = DatasetCollector()

    # Buka webcam
    cap = cv2.VideoCapture(2)

    # Buat windows
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Binary Result', cv2.WINDOW_NORMAL)

    # Set ukuran window
    cv2.resizeWindow('Original', 600, 400)
    cv2.resizeWindow('Contours', 600, 400)
    cv2.resizeWindow('Binary Result', 400, 600)

    print("\nControls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save 50 copies of the card")
    print("- Press 'h' to show valid card labels")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi kartu
        card_found, corners, contour_preview, edges = detect_card(frame)

        # Tampilkan original frame
        cv2.imshow('Original', frame)

        # Tampilkan contour preview
        cv2.imshow('Contours', contour_preview)

        # Tampilkan hasil warped jika kartu terdeteksi
        if card_found and corners is not None:
            warped, binary_warped = get_warped_card(frame, corners)
            # Invert binary result agar kartu putih
            binary_warped = cv2.bitwise_not(binary_warped)
            cv2.imshow('Binary Result', binary_warped)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Tampilkan template label yang valid
                print_valid_labels()
                label = input(
                    "\nEnter card label (e.g., 'queen_of_diamonds'): ")

                # Simpan 50 copy dengan variasi
                saved_files = collector.save_card_copies(
                    warped, binary_warped, label)
                if saved_files:
                    print(f"\nSaved {len(saved_files)} variations of the card")
                    print(f"Base filename: {saved_files[0]}")
            elif key == ord('h'):
                print_valid_labels()
            elif key == ord('q'):
                break
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
