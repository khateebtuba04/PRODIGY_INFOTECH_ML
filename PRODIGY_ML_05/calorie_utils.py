# Calorie estimates per 100g or per serving (approximate)

CALORIE_INFO = {
    'apple_pie': {'calories': 237, 'unit': '1 piece (1/8 of 9" pie)'},
    'baby_back_ribs': {'calories': 361, 'unit': '1 serving (3 oz)'},
    'baklava': {'calories': 334, 'unit': '1 piece (2 oz)'},
    'beef_carpaccio': {'calories': 250, 'unit': '1 serving (approx)'},
    'beef_tartare': {'calories': 420, 'unit': '1 serving (approx)'},
    'beet_salad': {'calories': 180, 'unit': '1 serving'},
    'beignets': {'calories': 240, 'unit': '1 beignet'},
    'bibimbap': {'calories': 560, 'unit': '1 bowl'},
    'bread_pudding': {'calories': 270, 'unit': '1/2 cup'},
    'breakfast_burrito': {'calories': 390, 'unit': '1 burrito'}
}

def get_calorie_info(class_name):
    return CALORIE_INFO.get(class_name, {'calories': 'Unknown', 'unit': '-'})
