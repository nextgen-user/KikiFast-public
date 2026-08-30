import time
from core.lcd_display import lcd_manager

print("Testing LCDDisplayManager...")
try:
    print("1. Status updates...")
    lcd_manager.update_status("Startup")
    time.sleep(2)

    lcd_manager.update_status("Wake word detected")
    time.sleep(2)

    lcd_manager.update_status("Listening", "what is your name?")
    time.sleep(2)

    lcd_manager.update_status("Thinking")
    time.sleep(2)

    print("2. Streaming text wrapping test...")
    long_response = "I am a voice companion robot running on a Raspberry Pi 5. Nice to meet you!"
    words = long_response.split()
    current = ""
    for word in words:
        current += word + " "
        lcd_manager.display_stream("Speaking:", current.strip())
        time.sleep(0.3)

    time.sleep(2)
    lcd_manager.update_status("Done speaking", "QA Time: 4.8s")
    time.sleep(3)

    lcd_manager.update_status("Idle")
    print("LCD Test complete.")

finally:
    # Make sure we leave it clear or idle
    lcd_manager.clear()
