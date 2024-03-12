from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import sys
import traceback
import time
import os


def initialize_driver(chrome_driver_path,url):
    try:
        options = Options()
        options.add_argument('--headless')
        service = Service(chrome_driver_path)
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url)
        driver.maximize_window()
    except Exception as e:
        print(f"Exception occured in initializing the driver {e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()

        # Get the formatted traceback as a string
        traceback_details = traceback.format_exception(exc_type, exc_value, exc_traceback)

        # Print the traceback details
        for line in traceback_details:
            print(line.strip())
        return None
    else:
        return driver


def login(chrome_driver_path,url,username,password):
    try:
        for _ in range(0,3):
            try:
                driver = initialize_driver(chrome_driver_path,url)
                if driver is not None:
                    user_name_xpath = '//input[@type="text" and @id="user_login"]'
                    user_name_box = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,user_name_xpath)))
                    user_name_box.send_keys(username)
                    print(f"User Name {username} entered")
                    time.sleep(3)
                    password_xpath = '//input[@type="password" and @id="user_pass"]'
                    password_box = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,password_xpath)))
                    password_box.send_keys(password)
                    print("Password entered")
                    time.sleep(2)
                    signin_xpath = '//input[@type="submit" and @id="wp-submit"]'
                    signin = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,signin_xpath)))
                    signin.click()
                    time.sleep(2)
                    print("Clicked on sign in")
                    time.sleep(4)
                    home_xpath = "//a[contains(text(),'Home')]"
                    home = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,home_xpath)))
                    if home:
                        print("Login success")
                        return True,driver
                    else:
                        raise Exception("Login Failed..Retrying")
                else:
                    raise Exception("Driver not found")
            except Exception as e:
                print(f"Excpetion occured trying again {e}")
        return False,None
    except Exception as e:
        print(f"Exception occured in login {e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()

        # Get the formatted traceback as a string
        traceback_details = traceback.format_exception(exc_type, exc_value, exc_traceback)

        # Print the traceback details
        for line in traceback_details:
            print(line.strip())
        return False,None


def get_data(driver,output_file_directory,links):
    try:
        # property_owners_xpath = "//a[contains(text(),'For Property Owners')]"
        # property_owners = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH,property_owners_xpath)))
        # property_owners_link = property_owners.get_attribute("href")
        output_file_paths = []
        for link in links:
            try:
                parts = link.split('/')
                file_name = parts[-2]
                output_file_path = os.path.join(output_file_directory,file_name)
                if '.txt' not in output_file_path:
                    output_file_path = output_file_path + '.txt'
                driver.get(link)
                property_text = driver.execute_script("return document.body.innerText;")
                print(property_text)
                with open(output_file_path, "w",encoding='utf-8') as file:
                    file.write(property_text)
            except Exception as e:
                print(f"Exception occured in extracting data for link {link} {e}")
            else:
                print(f"Successfully extracted data for link-{link}")
                output_file_paths.append(output_file_path)
    except Exception as e:
        print(f"Exception occured in getting data {e}")
        exc_type, exc_value, exc_traceback = sys.exc_info()

        # Get the formatted traceback as a string
        traceback_details = traceback.format_exception(exc_type, exc_value, exc_traceback)

        # Print the traceback details
        for line in traceback_details:
            print(line.strip())
        return False,[]
    else:
        return True,output_file_paths
