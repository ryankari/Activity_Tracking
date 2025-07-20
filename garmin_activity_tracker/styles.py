modern_button_style = """
QPushButton {
    background-color: #1976D2;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
    border: none;
}
QPushButton:hover {
    background-color: #1565C0;
}
QPushButton:pressed {
    background-color: #0D47A1;
}
"""

modern_combobox_style = """
QComboBox {
    background-color: #f5f5f5;
    color: #222;
    border-radius: 6px;
    padding: 6px 18px 6px 8px;
    font-size: 15px;
    border: 1px solid #1976D2;
}
QComboBox::drop-down {
    border: none;
    background: transparent;
}
QComboBox::down-arrow {
    image: url(:/qt-project.org/styles/commonstyle/images/arrowdown-16.png);
    width: 16px;
    height: 16px;
}
QComboBox QAbstractItemView {
    background: #fff;
    selection-background-color: #1976D2;
    selection-color: #fff;
    border-radius: 6px;
}
"""

ai_running_style = """
QPushButton {
    background-color: #FFD600;
    color: #222;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: bold;
    border: none;
}
"""