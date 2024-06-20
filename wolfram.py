import requests
API_KEY = 'YQ675G-3YV8WGLP49'
wolfram_code = 'Plot[Sin[x], {x, 0, 2*Pi}]'
url = 'http://api.wolframalpha.com/v2/query'

params = {
    'appid': API_KEY,
    'input': wolfram_code,
    'output': 'image' 
}

try:
    response = requests.get(url, params=params)

    if response.status_code == 200:
        image_data = response.content
        
        with open('wolfram_plot.png', 'wb') as f:
            f.write(image_data)
        print("Plot saved as 'wolfram_plot.png'")
    
    else:
        print("Request failed with status code:", response.status_code)
        print("Response text:", response.text)

except requests.exceptions.RequestException as e:
    print("Request exception:", e)
