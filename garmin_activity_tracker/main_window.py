"""
Filename: main_window.py
Description: Main window for Garmin Activity Tracker - UI layout and interactions.
Author: Ryan Kari
License: MIT
Created: 2025-07-20
"""

import sys
import os

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QLineEdit,
    QLabel,
    QSizePolicy,
    QMessageBox,
    QSpacerItem,
)
from PyQt5.QtWidgets import QComboBox, QHBoxLayout,QApplication, QInputDialog
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont

import datetime
import calendar
import shutil
import pandas as pd


from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib

import json
import importlib.resources

from garmin_activity_tracker.workers import LoadDataWorker, AIWorker

matplotlib.use("Qt5Agg")
from garmin_activity_tracker.garmin_core import ActivityTracker
from garmin_activity_tracker.plots import (
    plot_TSS,
    create_basic_metric_plot,
    createAdvancedMetricPlot,
    plotSplits,
    plot_calendar,
    plot_race_performance
)
from garmin_activity_tracker.utils_AI import get_response, AI_format, get_ollama_models
from garmin_activity_tracker.styles import (
    modern_button_style,
    modern_combobox_style,
    ai_running_style,
)




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Get screen size and DPI
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        dpi = screen.logicalDotsPerInch()

        # Calculate scaling factor based on DPI (96 DPI is standard)
        scale_factor = dpi / 96.0

        # Dynamically set window size (e.g., 80% of screen width and height)
        width = int(screen_geometry.width() * 0.8)
        height = int(screen_geometry.height() * 0.8)
        self.resize(width, height)

        # Center the window on the screen
        self.move(
            (screen_geometry.width() - width) // 2,
            (screen_geometry.height() - height) // 2,
        )
        self.config = None

        def load_config(filename="config_information.json"):
            try:
                # If bundled in the package
                with importlib.resources.open_text("garmin_activity_tracker", filename, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
                return {}
            
        if self.config is None:
            self.config = load_config() 

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- Left: Button column ---
        button_widget = QWidget()
        button_layout = QVBoxLayout(button_widget)
        button_layout.setAlignment(Qt.AlignTop)
        main_layout.addWidget(button_widget)

        # Add plot buttons
        self.plot_buttons = {}
        for label, func in [
            ("Basic Metrics", self.plot_basic_metrics),
            ("Advanced Metrics", self.plot_advanced_metrics),
            ("TSS", self.plot_tss),
            ("Plot Races", self.plot_races),
            ("Calendar", self.plot_calendar_func),
            ("Sync Latest Data", self.sync_latest_data),
            ("Back to Calendar", self.back_to_calendar),
        ]:
            btn = QPushButton(label)
            btn.setStyleSheet(modern_button_style)
            btn.clicked.connect(func)
            btn.setEnabled(False)
            button_layout.addWidget(btn)
            self.plot_buttons[label] = btn

        # Enable the Sync button immediately so users can always try to sync
        self.plot_buttons["Sync Latest Data"].setEnabled(True)
        self.plot_buttons["Back to Calendar"].hide()

        year_month_widget = QWidget()
        year_month_layout = QHBoxLayout(year_month_widget)
        year_month_layout.setAlignment(Qt.AlignLeft)

        # Year selector
        current_year = datetime.datetime.now().year
        years = list(range(current_year - 10, current_year + 1))
        self.year_combo = QComboBox()
        for y in years:
            self.year_combo.addItem(str(y))
        self.year_combo.setCurrentText(str(current_year))
        self.year_combo.currentIndexChanged.connect(self.update_calendar_from_selector)
        self.year_combo.setStyleSheet(modern_combobox_style)

        year_month_layout.addWidget(QLabel("Year:"))
        year_month_layout.addWidget(self.year_combo)

        # Month selector
        self.month_combo = QComboBox()
        self.month_combo.setStyleSheet(modern_combobox_style)
        for i, month in enumerate(calendar.month_abbr[1:], 1):  # Jan to Dec
            self.month_combo.addItem(month, i)
        self.month_combo.setCurrentIndex(datetime.datetime.now().month - 1)
        self.month_combo.setMaxVisibleItems(12) 
        self.month_combo.currentIndexChanged.connect(self.update_calendar_from_selector)
        year_month_layout.addWidget(QLabel("Month:"))
        year_month_layout.addWidget(self.month_combo)

        button_layout.addWidget(year_month_widget)
        button_layout.addItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

        btn = QPushButton("Chat with AI Coach")
        btn.setStyleSheet(modern_button_style)
        btn.clicked.connect(self.initiateAIChat)
        button_layout.addWidget(btn)
        self.plot_buttons["Chat with AI Coach"] = btn

        self.last_calendar_year = int(self.year_combo.currentText())
        self.last_calendar_month = self.month_combo.currentData()
        self.last_calendar_mode = "last28"

        # In your __init__ after other widgets
        if self.is_ollama_installed():
            models = get_ollama_models(self)
            if models:
                self.ollama_model_combo = QComboBox()
                self.ollama_model_combo.addItems(models)
                self.ollama_model_combo.setStyleSheet(modern_combobox_style)
                button_layout.addWidget(QLabel("Ollama Model:"))
                button_layout.addWidget(self.ollama_model_combo)
                self.selected_ollama_model = models[0]
                self.ollama_model_combo.currentTextChanged.connect(
                    self.set_ollama_model
                )
            else:
                button_layout.addWidget(QLabel("No Ollama models found."))
        else:
            button_layout.addWidget(QLabel("Ollama not installed."))

        # --- Center: Plot area ---
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)
        plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(plot_widget, stretch=1)

        self.figure = plt.Figure(figsize=(9, 6))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, stretch=5)  # Larger plot area

        # --- Bottom: AI Chat Console ---
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        dpi = QApplication.primaryScreen().logicalDotsPerInch()
        scale = dpi / 96.0  # 96 DPI is standard

        font = QFont()
        font.setPointSizeF(10 * scale)  # Base size 10, scaled
        self.console.setFont(font)

        self.conversation_history = []

        self.input_line = QLineEdit()
        self.input_line.setPlaceholderText("Type your message to the AI coach...")
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.continueAIChat)

        chat_layout = QHBoxLayout()
        chat_layout.addWidget(self.input_line)
        chat_layout.addWidget(self.send_button)

        plot_layout.addWidget(QLabel("AI Coach Console:"))
        plot_layout.addWidget(self.console, stretch=1)
        plot_layout.addLayout(chat_layout)

        # --- Data ---
        self.tracker = None
        self.activities = {}
        self.activities['All'] = {None}
        self.activities["Running"] = {'Summary':None, 'Splits':None, 'TSS':None}
        self.activities["Cycling"] = {'Summary':None, 'Splits':None, 'TSS':None}
        #self.df_summary = None
        #self.df_running = None
        #self.df_running_splits = None
        # Note: TSS is now stored in self.activities['Running']['TSS']
        print(get_ollama_models(self))
        self.load_data_async(use_api=False)


    def is_ollama_installed(self):
        return shutil.which("ollama") is not None
    
    def get_garmin_credentials(self):
        username = os.getenv("GARMIN_USERNAME")
        password = os.getenv("GARMIN_PASSWORD")
        if not username:
            username, ok = QInputDialog.getText(self, "Garmin Username", "Enter your Garmin username:")
            if not ok or not username:
                return None, None
        if not password:
            password, ok = QInputDialog.getText(self, "Garmin Password", "Enter your Garmin password:", QLineEdit.Password)
            if not ok or not password:
                return None, None
        return username, password



    def set_ollama_model(self, model_name):
        self.selected_ollama_model = model_name

    def sync_latest_data(self):
        self.plot_buttons["Sync Latest Data"].setStyleSheet(ai_running_style)
        QApplication.processEvents()
        self.load_data_async(use_api=True)

    def load_data_async(self, use_api=False):
        if use_api:
            username, password = self.get_garmin_credentials()
            if not username or not password:
                QMessageBox.critical(
                    self, "Credentials Error",
                    "GARMIN_USERNAME or GARMIN_PASSWORD environment variable not set."
                )
                return
            self.tracker = ActivityTracker(username, password, self.config)
        else:
            # Credentials not needed for local-only mode
            self.tracker = ActivityTracker(None, None, self.config)

        self.console.append("Use Garmin API = {}".format(use_api))
        self.load_worker = LoadDataWorker(self.tracker, use_api=use_api,config=self.config)
        self.load_worker.progress.connect(self.console.append)
        self.load_worker.finished.connect(self.handle_data_loaded)
        self.load_worker.enable_sync_button.connect(self.enable_sync_button)
        self.load_worker.start()


    def handle_data_loaded(self, activities):
        if activities is None:
            self.console.append("Data load failed.")
            return
        self.activities = activities

        # Check if we have valid data
        if activities['Running']['Summary'] is None or activities['Running']['Summary'].empty:
            self.console.append("No running data available.")
            return

        if activities['Running']['Splits'] is None:
            self.console.append("Warning: No splits data available.")
            self.activities['Running']['Splits'] = None  # Initialize as empty DataFrame
        
        # Calculate and store TSS in the activities structure
        self.activities['Running']['TSS'] = self.tracker.calculate_tss(self.activities['Running']['Summary'], self.config)
        self.plot_calendar_func(year=None, month=None)  # Default to last 28 days
        self.plot_buttons["Sync Latest Data"].setStyleSheet(modern_button_style)
        for btn in self.plot_buttons.values():
            btn.setEnabled(True)

    def enable_sync_button(self):
        """Re-enable the sync button when called from worker thread"""
        self.plot_buttons["Sync Latest Data"].setEnabled(True)
        self.plot_buttons["Sync Latest Data"].setStyleSheet(modern_button_style)

    def clear_figure(self):
        self.figure.clear()
        self.canvas.draw()

    def plot_tss(self):
        self.clear_figure()
        ax = self.figure.add_subplot(111)
        plot_TSS(self.activities['Running']['Summary'], self.activities['Running']['TSS'], ax)
        self.canvas.draw()

    def plot_basic_metrics(self):
        self.clear_figure()
        axes = self.figure.subplots(3, 1, sharex=True)  # 3 rows, 1 column
        create_basic_metric_plot(self.activities['Running']['Summary'], axes, show_trend=True)
        self.canvas.draw()

    def plot_advanced_metrics(self):
        self.clear_figure()
        axes = self.figure.subplots(3, 1, sharex=True)  # 3 rows, 1 column
        createAdvancedMetricPlot(self.activities['Running']['Summary'], axes, show_trend=True)
        self.canvas.draw()

    def plot_splits(self):
        self.clear_figure()
        ax = self.figure.add_subplot(111)
        
        splits = self.activities['Running']['Splits']
        if splits is None or splits.empty:
            ax.text(
                0.5, 0.5, "No splits data available.\nTry syncing with Garmin API to download splits.", 
                ha="center", va="center"
            )
            self.plot_buttons["Back to Calendar"].show()
            self.canvas.draw()
            return
            
        plotSplits(splits, self.activities['Running']['Summary'], ax, self.config)
        self.plot_buttons["Back to Calendar"].show()
        self.canvas.draw()

    def plot_splits_for_activity(self, activity_id):
        self.clear_figure()
        ax = self.figure.add_subplot(111)
        
        # Check if splits data is available and has the required column
        splits = self.activities['Running']['Splits']
        if splits is None or splits.empty:
            ax.text(
                0.5, 0.5, "No splits data available.\nTry syncing with Garmin API to download splits.", 
                ha="center", va="center"
            )
            self.plot_buttons["Back to Calendar"].show()
            self.canvas.draw()
            return
            
        if 'activityId' not in splits.columns:
            ax.text(
                0.5, 0.5, "Splits data missing activityId column.\nTry syncing with Garmin API.", 
                ha="center", va="center"
            )
            self.plot_buttons["Back to Calendar"].show()
            self.canvas.draw()
            return
        
        # Filter splits for the selected activity
        splits = splits[splits["activityId"].astype(str) == str(activity_id)]
        activity_summary = self.activities['Running']['Summary'][
        self.activities['Running']['Summary']["activityId"].astype(str) == str(activity_id)
        ]

        self.plot_buttons["Back to Calendar"].show()
        
        if activity_summary.empty:
            ax.text(
                0.5, 0.5, "No splits found for this activity.", ha="center", va="center"
            )
        else:
            plotSplits(splits, activity_summary, ax, activityId=activity_id, config=self.config)
        self.canvas.draw()

    def plot_races(self):
        self.clear_figure()
        ax = self.figure.add_subplot(111)
        plot_race_performance(self.activities['Running']['Summary'], ax, self.config)
        self.canvas.draw()
        

    def plot_calendar_func(self, year=None, month=None):
        if year is not None and month is not None:
            self.last_calendar_year = year
            self.last_calendar_month = month
            self.last_calendar_mode = "month"
        else:
            self.last_calendar_mode = "last28"

        self.clear_figure()
        ax = self.figure.add_subplot(111)

        self.calendar_dates = plot_calendar(self.activities, ax, self.config, year=year, month=month)
        self.canvas.draw()
        self.canvas.mpl_connect("button_press_event", self.on_calendar_click)
        self.plot_buttons["Back to Calendar"].hide()

    def update_calendar_from_selector(self):
        year = int(self.year_combo.currentText())
        month = self.month_combo.currentData()
        self.plot_calendar_func(year=year, month=month)

    def back_to_calendar(self):
        if getattr(self, "last_calendar_mode", "last28") == "month":
            # Restore the last viewed calendar month/year
            self.year_combo.setCurrentText(str(self.last_calendar_year))
            # Find the correct index for the month
            for i in range(self.month_combo.count()):
                if self.month_combo.itemData(i) == self.last_calendar_month:
                    self.month_combo.setCurrentIndex(i)
                    break
            self.plot_calendar_func(
                year=self.last_calendar_year, month=self.last_calendar_month
            )
        else:
            # Restore the last 28 days view
            self.plot_calendar_func(year=None, month=None)

    def on_calendar_click(self, event):
        # Only respond if the calendar is currently shown
        if not hasattr(self, "calendar_dates") or self.calendar_dates is None:
            return
    
        if event.inaxes and self.calendar_dates:   
            x, y = int(round(event.xdata)), int(round(event.ydata))
            clicked_date = self.calendar_dates.get((x, y))
            if clicked_date:
                activities = self.activities['Running']['Summary'][self.activities['Running']['Summary'].index.date == clicked_date]
                if not activities.empty:
                    if len(activities) == 1:
                        activity_id = activities.iloc[0]["activityId"]
                    else:
                        # Let user pick which activity

                        item, ok = QInputDialog.getItem(
                            self,
                            "Select Activity",
                            "Multiple activities found for this date. Choose one:",
                            activities['activityName'].values,
                            0,
                            False,
                        )
                        if not ok or not item:
                            return
                        # Extract activityId from the selected string
                        activity_id = activities[activities['activityName'] == item]['activityId'].values[0]
                    print(activity_id)
                    self.plot_splits_for_activity(activity_id)



    def initiateAIChat(self):
        print("Initiating AI Chat")
   
        # Clear console and chat history
        self.console.clear()
        self.conversation_history = []
        user_msg = self.input_line.text().strip()
        self.conversation_history.append(f"User: {user_msg}")

        # Prepare prompt for AI
        prompt_content, _ = AI_format(self.activities, self.config)
        if prompt_content.startswith("ERROR:"):
            self.console.append(prompt_content)  # Or however you display messages
            return  # Exit the function, don't proceed with AI call
        formatted_response = (
            f'<div style="white-space: pre-wrap;">{prompt_content}</div>'
        )
        self.console.append(f"<b>You:</b> {formatted_response}")
        try:
            if self.config.get("AI_format", {}).get("create_prompt_only", True):
                    # Copy prompt_content to clipboard
                clipboard = QApplication.clipboard()
                clipboard.setText(prompt_content)
                self.console.append("<i>Prompt copied to clipboard.</i>")
                return
            
            self.plot_buttons["Chat with AI Coach"].setStyleSheet(ai_running_style)
            QApplication.processEvents()
            ai_response = ""

            model = getattr(self, "selected_ollama_model", None)

            self.ai_worker = AIWorker(prompt_content, model)
            self.ai_stream_buffer = ""  # Buffer for the streaming response
            self.console.append(f"<b>AI Coach:</b><br>")
            self.ai_worker.chunk_ready.connect(self.append_ai_chunk)
            self.ai_worker.finished.connect(self.handle_ai_response)
            self.ai_worker.start()

        except Exception as e:
            self.console.append(f"<b>Error:</b> {e}")


    def continueAIChat(self):
        user_msg = self.input_line.text().strip()
        if not user_msg:
            return
        self.console.append(f"<b>You:</b> {user_msg}")
        self.conversation_history.append(f"User: {user_msg}")
        # Prepare prompt for AI
        prompt_content, _ = AI_format(self.activities, self.config)
        prompt_content += "\n" + "\n".join(self.conversation_history)
        try:
            self.plot_buttons["Chat with AI Coach"].show()
            self.plot_buttons["Chat with AI Coach"].setStyleSheet(ai_running_style)
            QApplication.processEvents()
            print("prompt content = \n", prompt_content)
            print("self.conversation_history = \n", self.conversation_history)

            model = getattr(self, "selected_ollama_model", None)

            # Start streaming AI worker thread
            self.ai_worker = AIWorker(prompt_content, model)
            self.ai_stream_buffer = ""  # Buffer for the streaming response
            self.console.append(f"<b>AI Coach:</b><br>")
            self.ai_worker.chunk_ready.connect(self.append_ai_chunk)
            self.ai_worker.finished.connect(self.handle_ai_response)
            self.ai_worker.start()


        except Exception as e:
            self.console.append(f"<b>Error:</b> {e}")

    def append_ai_chunk(self, text):
        self.ai_stream_buffer += text

        cursor = self.console.textCursor()
        cursor.movePosition(cursor.End)
        self.console.setTextCursor(cursor)
        self.console.insertPlainText(text)
        self.console.ensureCursorVisible()
        QApplication.processEvents()

    def handle_ai_response(self, ai_response):
        self.conversation_history.append(f"AI Coach: {ai_response}")
        self.input_line.clear()
        self.plot_buttons["Chat with AI Coach"].setStyleSheet(modern_button_style)
        #self.plot_buttons["Chat with AI Coach"].hide()


def run_app():
    import sys
    from PyQt5.QtWidgets import QApplication
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    screen = app.primaryScreen()
    dpi = screen.logicalDotsPerInch()  # Typically 96 for normal screens

    # Set base font size relative to 96 dpi
    scale_factor = dpi / 96.0
    font = app.font()
    font.setPointSizeF(10 * scale_factor)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


