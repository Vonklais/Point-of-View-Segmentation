from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загрузка обученной модели YOLO для сегментации
model = YOLO("yolov8m-seg.pt")  # Убедитесь, что у вас есть подходящая модель для сегментации

# Открытие видеопотока
cap = cv2.VideoCapture("Test636.mp4")

# Загрузка координат из CSV файла
coords_df = pd.read_csv("Test636_coords.csv")

# Получение ширины и высоты экрана
screen_width = 1000  # Задайте ширину экрана
screen_height = 900  # Задайте высоту экрана

# Получение ширины и высоты видео
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Коэффициент масштабирования для уменьшения размера видео
scale_width = screen_width / video_width
scale_height = screen_height / video_height
scale = min(scale_width, scale_height) * 0.9  # Дополнительное уменьшение для удобства

# Новые размеры видео
new_width = int(video_width * scale)
new_height = int(video_height * scale)

# Настройка видеозаписи
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Test636_new.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (new_width, new_height))

# Флаг для проверки паузы
paused = False

# Функция для обработки событий мыши
def on_mouse(event, x, y, flags, param):
    global paused
    if event == cv2.EVENT_LBUTTONDOWN:
        paused = not paused

# Установка функции обработки событий мыши
cv2.namedWindow("Video")
cv2.setMouseCallback("Video", on_mouse)

# Словарь для отслеживания времени на каждом классе
class_time = {}
total_frames = 0

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        # Получение текущего номера кадра
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        total_frames += 1

        # Выполнение предсказания на текущем кадре
        results = model(frame)

        # Получение координат для текущего кадра
        if current_frame in coords_df['frame'].values:
            x_coord = coords_df.loc[coords_df['frame'] == current_frame, 'x'].values[0]
            y_coord = coords_df.loc[coords_df['frame'] == current_frame, 'y'].values[0]

            # Проверка, находится ли точка взора внутри масок и определение маски, которая должна быть отображена
            mask_to_display = None
            label_to_display = None
            min_area = float('inf')
            found_object = False

            for result in results:
                masks = result.masks
                if masks is not None and len(masks) > 0:
                    for i, mask in enumerate(masks.xy):
                        if len(mask) == 0:
                            continue
                        x1, y1, x2, y2 = min(mask[:,0]), min(mask[:,1]), max(mask[:,0]), max(mask[:,1])
                        area = (x2 - x1) * (y2 - y1)
                        if x1 - 10 <= x_coord <= x2 + 10 and y1 - 10 <= y_coord <= y2 + 10:
                            found_object = True
                            if area < min_area:
                                min_area = area
                                mask_to_display = mask
                                label_to_display = result.names[int(result.boxes[i].cls.item())]

            if found_object:
                # Обновление времени для класса
                if label_to_display in class_time:
                    class_time[label_to_display] += 1
                else:
                    class_time[label_to_display] = 1

            # Отображение только нужной маски
            if mask_to_display is not None:
                mask_image = np.zeros_like(frame)
                cv2.fillPoly(mask_image, [mask_to_display.astype(int)], (0, 0, 255))
                frame = cv2.addWeighted(frame, 1, mask_image, 0.5, 0)

                x1, y1, x2, y2 = min(mask_to_display[:,0]), min(mask_to_display[:,1]), max(mask_to_display[:,0]), max(mask_to_display[:,1])

                # Рисование рамки вокруг маски
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, label_to_display, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Отрисовка круга в координатах (x_coord, y_coord)
            cv2.circle(frame, (int(x_coord), int(y_coord)), 10, (0, 0, 255), 2)

        # Изменение размера кадра
        frame = cv2.resize(frame, (new_width, new_height))

        # Запись кадра в выходной файл
        out.write(frame)

    # Показ кадра с обнаруженными объектами и кругом
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Вычисление доли времени для каждого класса
class_percentage = {label: (time / total_frames) * 100 for label, time in class_time.items()}

# Фильтрация классов с долей времени меньше 0.1%
class_percentage = {label: percentage for label, percentage in class_percentage.items() if percentage >= 0.1}

# Сортировка классов по убыванию доли времени
sorted_class_percentage = dict(sorted(class_percentage.items(), key=lambda item: item[1], reverse=True))

# Построение диаграммы в виде вертикальных столбцов
labels = list(sorted_class_percentage.keys())
sizes = list(sorted_class_percentage.values())

fig, ax = plt.subplots()
ax.bar(labels, sizes, color='blue')
ax.set_xlabel('Классы объектов')
ax.set_ylabel('Доля времени (%)')
ax.set_title('Доля времени, направленная на объекты определённых классов')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()













