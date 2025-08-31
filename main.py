from playwright.sync_api import sync_playwright
import time
from bs4 import BeautifulSoup

p = sync_playwright().start()

browser = p.chromium.launch(headless=True)

context = browser.new_context(user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36")

page = context.new_page()

page.goto("https://www.wanted.co.kr/wdlist")
time.sleep(3)

page.wait_for_selector("button.Aside_searchButton__Ib5Dn", timeout=10000)
page.click("button.Aside_searchButton__Ib5Dn")
time.sleep(3)

page.get_by_placeholder("검색어를 입력해 주세요.").fill("AI")

time.sleep(3)

page.keyboard.down("Enter")

time.sleep(10)

page.click("a#search_tab_position")

for x in range(5):
    time.sleep(3)
    page.keyboard.down("End")

time.sleep(3)

content = page.content()

soup = BeautifulSoup(content, "html.parser")

page.screenshot(path="screenshot.png")