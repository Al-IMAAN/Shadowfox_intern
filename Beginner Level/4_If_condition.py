# ---------------------------------------------------
# 4.If Condition
print('---------- 4. If Condition -----------')
# ---------------------------------------------------

# 4.1 BMI Calculator
print('---------- BMI Caluculator ----------')

height = float(input('Enter height in meters : '))
weight = float(input('Enter weight in Kilograms : '))

bmi = weight / (height ** 2)

if bmi >= 30:
    print('Obesity')
elif 25 <= bmi <= 29:
    print('Over Weight')
elif 18.5 <= bmi < 25:
    print('Normal')
else:
    print('Under Weight')

# -----------------------------------------------------
# 4.2 Find States in the country
print('----------- Find States belongs to Country -----------')

australia = ['Sydney', 'Melbourne', 'Brisbane', 'Perth']
uae = ['Dubai', 'Abu Dhabi', 'Sharjah', 'Ajman']
india = ['Mumbai', 'Bangalore', 'Chennai', 'Delhi']

city = input('Enter a city name : ').title()

if city in australia:
    print(f'{city} is in Australia')
elif city in uae:
    print(f'{city} is in UAE')
elif city in india:
    print(f'{city} is in India')
else:
    print(f'{city} is not in the list')

# ---------------------------------------------------
# 4.3 Check whether the two cities belongs to same country.
print('------ Check whether the two cities belongs to same country ------')


def get_country(state):
    if state in australia:
        return 'Australia'
    elif state in uae:
        return 'UAE'
    elif state in india:
        return 'India'
    else:
        return None


city1 = input('Enter the First city name : ').title()
city2 = input('Enter the Second city name : ').title()

country1 = get_country(city1)
country2 = get_country(city2)

if country1 and country2:
    if country1 == country2:
        print(f'Both {city1} and {city2} are in {country1}.')
    else:
        print(f'{city1} is in {country1} and {city2} is in {country2}.')
else:
    if not country1:
        print(f'{city1} is not in the  list.')
    if not country2:
        print(f'{city2} is not in the list.')
