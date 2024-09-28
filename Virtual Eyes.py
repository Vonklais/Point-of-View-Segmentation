from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dearpygui.dearpygui as dpg
import os
import sys

# Определение пути к модели
if getattr(sys, 'frozen', False):
    # Если программа упакована в exe
    base_path = os.path.dirname(sys.executable)
else:
    # Если программа запускается в исходном виде
    base_path = os.path.dirname(os.path.abspath(__file__))

model_file = os.path.join(base_path, 'models', 'yolov8m-seg.pt')


# Глобальные переменные для хранения путей к файлам
video_file = None
coords_file = None
output_directory = None
output_video_name = "output.mp4"  # Добавлена переменная для имени выходного файла

# Глобальная модель и данные
coords_df = None


def process_video():
    if not video_file or not coords_file or not output_directory:
        dpg.show_item("error_window")
        return

    model = YOLO(model_file)
    model.to("cuda")
    cap = cv2.VideoCapture(video_file)
    coords_df = pd.read_csv(coords_file)

    # Получение ширины и высоты видео
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Настройка видеозаписи
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(output_directory, output_video_name)
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (video_width, video_height))

    # Словарь для отслеживания времени на каждом классе
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Общее количество кадров
    class_time = {}  # Инициализация переменной class_time

    dpg.set_value("progress_bar", 0.0)
    dpg.configure_item("status_text", default_value="Processing... 0%")

    coords_dict = coords_df.set_index('frame').to_dict(orient='index')

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            break
        print(f"Using device: {model.device}")
        # Получение текущего номера кадра
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Выполнение предсказания на текущем кадре
        results = model(frame)

        # Обновление прогресс-бара
        if current_frame % 10 == 0:
            progress = current_frame / total_frames
            dpg.set_value("progress_bar", progress)
            dpg.configure_item("status_text", default_value=f"Processing... {int(progress * 100)}%")

        # Получение координат для текущего кадра
        if current_frame in coords_dict:
            x_coord = coords_dict[current_frame]['x']
            y_coord = coords_dict[current_frame]['y']

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
                            print(f"НЕТ ОБЪЕКТОВ")
                            continue

                        x1, y1, x2, y2 = min(mask[:, 0]), min(mask[:, 1]), max(mask[:, 0]), max(mask[:, 1])
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

                x1, y1, x2, y2 = min(mask_to_display[:, 0]), min(mask_to_display[:, 1]), max(
                    mask_to_display[:, 0]), max(mask_to_display[:, 1])

                # Рисование рамки вокруг маски
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, label_to_display, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 255), 2)

            # Отрисовка круга в координатах (x_coord, y_coord)
            cv2.circle(frame, (int(x_coord), int(y_coord)), 10, (0, 0, 255), 2)

        # Изменение размера кадра
        if video_width != frame.shape[1] or video_height != frame.shape[0]:
            frame = cv2.resize(frame, (video_width, video_height))

        # Запись кадра в выходной файл
        out.write(frame)

    cap.release()
    out.release()

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
    plt.savefig(os.path.join(output_directory, 'class_distribution.png'))

    dpg.set_value("progress_bar", 1.0)
    dpg.configure_item("status_text", default_value=f"Process completed! Video saved to {output_directory}")


# Функции загрузки файлов
def select_video_file(sender, app_data):
    global video_file
    video_file = app_data['file_path_name']
    dpg.set_value("video_file_text", video_file)


def select_coords_file(sender, app_data):
    global coords_file
    coords_file = app_data['file_path_name']
    dpg.set_value("coords_file_text", coords_file)


def select_output_directory(sender, app_data):
    global output_directory
    output_directory = app_data['file_path_name']
    dpg.set_value("output_directory_text", output_directory)


# Функция для установки имени выходного файла
def set_output_video_name(sender, app_data):
    global output_video_name
    output_video_name = app_data


# Функция для обновления директории при изменении текста
def update_output_directory(sender, app_data):
    global output_directory
    output_directory = app_data


# Функция для обновления расположения кнопок при изменении размера окна
def update_button_positions():
    width = dpg.get_viewport_width()
    button_width = 200
    center_x = (width - button_width) / 2

    dpg.set_item_pos("video_button", (center_x, 50))
    dpg.set_item_pos("video_file_text", (center_x, 100))
    dpg.set_item_pos("csv_button", (center_x, 130))
    dpg.set_item_pos("coords_file_text", (center_x, 180))
    dpg.set_item_pos("dir_button", (center_x, 210))
    dpg.set_item_pos("output_directory_text", (center_x, 260))
    dpg.set_item_pos("output_video_name_text", (center_x, 310))
    dpg.set_item_pos("process_button", (center_x, 360))
    dpg.set_item_pos("status_text", (center_x, 410))


# Создание контекста
dpg.create_context()

# Настройка интерфейса
dpg.create_viewport(title='Video Processor', width=800, height=600, resizable=True, decorated=True)

# Создание основного окна приложения
with dpg.handler_registry():
    dpg.add_key_release_handler(key=dpg.mvKey_Escape, callback=lambda: dpg.stop_dearpygui())

with dpg.window(label="Video Processor", pos=(0, 0), width=800, height=600, no_title_bar=True, no_resize=True,
                no_move=True):
    # Стиль для кнопок
    button_width = 200
    button_height = 40

    # Кнопки
    video_button = dpg.add_button(label="Choose Video File", callback=lambda: show_file_dialog("video_file_dialog"),
                                  width=button_width, height=button_height, tag="video_button", pos=(0, 50))
    dpg.add_text("", tag="video_file_text", pos=(0, 100), color=(255, 255, 255))

    csv_button = dpg.add_button(label="Choose CSV File", callback=lambda: show_file_dialog("coords_file_dialog"),
                                width=button_width, height=button_height, tag="csv_button", pos=(0, 130))
    dpg.add_text("", tag="coords_file_text", pos=(0, 180), color=(255, 255, 255))

    dir_button = dpg.add_button(label="Choose Directory", callback=lambda: show_file_dialog("output_directory_dialog"),
                                width=button_width, height=button_height, tag="dir_button", pos=(0, 210))

    # Поле ввода для директории с обработчиком
    dpg.add_input_text(tag="output_directory_text", width=button_width, default_value="", pos=(0, 260),
                       callback=update_output_directory)

    # Поле ввода имени выходного видеофайла
    dpg.add_input_text(label="Output Video Name", default_value="output.mp4", callback=set_output_video_name,
                       pos=(0, 310), width=button_width, tag="output_video_name_text")

    # Кнопка начала обработки
    dpg.add_button(
        label="Start Processing",
        callback=lambda: process_video(),
        pos=(0, 360),
        width=button_width,
        height=button_height,
        tag="process_button"
    )

    dpg.add_text("", tag="status_text", pos=(0, 410))
    # Добавление прогресс-бара
    dpg.add_progress_bar(tag="progress_bar", default_value=0.0, pos=(200, 440), width=400)


# Функция для показа диалогов
def show_file_dialog(dialog_id):
    # Скрываем все диалоги перед открытием нового
    dpg.hide_item("video_file_dialog")
    dpg.hide_item("coords_file_dialog")
    dpg.hide_item("output_directory_dialog")

    # Показываем нужный диалог
    dpg.show_item(dialog_id)


# Создание диалогов
with dpg.file_dialog(directory_selector=False, show=False, callback=select_video_file, id="video_file_dialog",
                     width=500, height=400):
    dpg.add_file_extension(".mp4")

with dpg.file_dialog(directory_selector=False, show=False, callback=select_coords_file, id="coords_file_dialog",
                     width=500, height=400):
    dpg.add_file_extension(".csv")

with dpg.file_dialog(directory_selector=True, show=False, callback=select_output_directory,
                     id="output_directory_dialog", width=500, height=400):
    pass

with dpg.window(label="Error", show=False, modal=True, id="error_window"):
    dpg.add_text("Please fill in all fields and try again.")
    dpg.add_button(label="OK", callback=lambda: dpg.hide_item("error_window"))

# Настройка и запуск интерфейса
dpg.setup_dearpygui()

# Обновление расположения кнопок при первоначальном запуске
update_button_positions()


# Обработчик изменения размера окна
def resize_handler(sender, app_data):
    update_button_positions()


dpg.set_viewport_resize_callback(resize_handler)

dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()







