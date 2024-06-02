function convertTemperature() {
    const temp_input = document.getElementById('temp');
    const conversion_type_input = document.getElementById('conversion-type');
    let result = 0, converted_temp = 0;

    const temp = parseFloat(temp_input.value);
    const conversion_type = conversion_type_input.value;

    if (conversion_type === 'CtoF') {
        converted_temp = (temp * 9/5) + 32;
        result = `${temp}°C is ${converted_temp.toFixed(2)}°F`;
    }
    else if (conversion_type === 'CtoK') {
        converted_temp = temp + 273.15;
        result = `${temp}°C is ${converted_temp.toFixed(2)}°K`;
    }

    else if (conversion_type === 'FtoC') {
        converted_temp = (temp - 32) * 5/9;
        result = `${temp}°F is ${converted_temp.toFixed(2)}°C`;
    }

    else if (conversion_type === 'FtoK'){
        converted_temp = (temp - 32) * 5/9 + 273.15;
        result = `${temp}°F is ${converted_temp.toFixed(2)}°K`;
    }

    else if (conversion_type === 'KtoC'){
        converted_temp = temp - 273.15;
        result = `${temp}°K is ${converted_temp.toFixed(2)}°C`;
    }

    else if (conversion_type === 'KtoF'){
        converted_temp = (temp - 273.15) * 9/5 + 32;
        result = `${temp}°K is ${converted_temp.toFixed(2)}°F`;
    }

    document.getElementsByName('result')[0].value = result;
}