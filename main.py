# main.py
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.metrics import dp
import os

class TestApp(App):
    def build(self):
        Window.size = (1536, 864)
        return MyBoxLayout()

class MyBoxLayout(BoxLayout):
    def __init__(self, **kwargs):
        super(MyBoxLayout, self).__init__(**kwargs)
        self.bind(on_parent=self.on_parent)

    def on_parent(self, instance, value):
        if value:
            left_box = self.ids.left_box
            if left_box:
                left_box.bind(on_size=self.update_size_hint_x)

    def update_size_hint_x(self, instance, value):
        left_box = self.ids.left_box
        if left_box:
            left_box.size_hint_x = instance.width / self.width

class LeftBox(BoxLayout):
    pass

class CenterBox(BoxLayout):
    pass

class RightBox(BoxLayout):
    def update_label_text(self, button_text):
        self.ids.right_label.text = f"{button_text}"

class MyAppButton(Button):
    def __init__(self, **kwargs):
        super(MyAppButton, self).__init__(**kwargs)
        self.bind(on_release=self.on_button_release)

    def on_button_release(self, instance):
        app = App.get_running_app()
        app.root.ids.right_box.update_label_text(self.text)

if __name__ == '__main__':
    TestApp().run()