exchange_rate = {'CNY': 6.9588, 'ZAR': 13.8691, 'KRW': 1120.456, 'SGD': 1.3718,
                 'THB': 32.9627, 'ARS': 37.7315, 'TWD': 30.8448, 'SAR': 3.752,
                 'USD': 1.0, 'MYR': 4.1797, 'SEK': 9.109, 'NZD': 1.4542,
                 'HKD': 7.8236, 'VND': 4.4e-05, 'IDR': 14330.4526,
                 'AUD': 1.3672, 'NOK': 8.5922, 'GBP': 0.7841, 'EUR': 0.8833,
                 'JPY': 113.4941, 'INR': 69.6475, 'PHP': 52.411, 'AED': 3.673,
                 'RUB': 67.0545, 'BHD': 0.377, 'CHF': 0.9984, 'OMR': 0.3851,
                 'UAH': 28.3135, 'CAD': 1.3296, 'TRY': 5.2118, 'PLN': 3.7887,
                 'ILS': 3.7211, 'PKR': 134.1847, 'DKK': 6.5934, 'RON': 4.1121,
                 'LKR': 179.3797, 'JOD': 0.7095, 'KWD': 0.3042, 'QAR': 3.6409,
                 'CZK': 22.9453, 'HUF': 285.9137, 'BRL': 3.8672,
                 'EGP': 17.9113, 'FJD': 2.1133, 'MXN': 20.3997, 'BDT': 84.1851,
                 'KZT': 375.1462, 'NGN': 365.7219, 'XPF': 105.4033,
                 'KHR': 0.00025, 'LAK': 8.6e-05}

def currency_convert_to_USD(value, currency):
    """
    Receives a money amount and a currency and converts it to USD
    :param value: The amount of money to convert
    :param currency: The original currency
    :return: The worth of this money in USD
    """
    if (currency not in exchange_rate):
        raise ValueError("The value of " + currency + " is not defined.")
    return value / exchange_rate[currency]

