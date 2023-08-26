#ruleaza 10 validari cu tm. layout = Pictures/layout_auto.png.
#import pydirectinput
import pyautogui
#import mouse
import win32api, win32con
import time

pos_ok = (1412, 424) #dupa validare apare un mesaj.
pos_validate = (1412, 325) #butonul pe care apas pt inceperea validarii.
#unde tp sa apas ai selectez un replay.
pos_replays = [(1533, 163), (1533, 194), (1533, 221), (1536, 254), (1538, 287), (1541, 312), (1541, 342), (1540, 373), (1537, 402), (1537, 432)]
pos_launch = (1429, 672) #pe ce apas ca sa dau launch la un replay.
pos_vscode = (790, 710) #highlight editor.
pos_tmf = (1859, 190) #highlight tmforever.

print("begin script.")
time.sleep(1)

def jaf(x, y):
    win32api.SetCursorPos((x, y))
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
    time.sleep(0.1)


#jaf = lambda x, y: pyautogui.moveTo(x, y, duration = 0.25); time.sleep(0.1); pyautogui.leftClick()
#jaf = lambda x, y, cntClicks = 1: mouse.move(x, y, duration = 0.5); mouse.click()
#jaf(*pos_vscode)
#pyautogui.write("py main.py\n")
#quit()

for i in range(10):
    jaf(*pos_vscode)
    pyautogui.write("py main.py\n")
    jaf(*pos_tmf)
    jaf(*pos_replays[i])
    jaf(*pos_launch)
    jaf(*pos_validate)
    time.sleep(7.5)
    jaf(*pos_ok)
    print(f"finished {i}.")
