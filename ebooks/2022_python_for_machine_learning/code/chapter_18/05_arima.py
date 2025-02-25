from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import pandas as pd

data = [266.0,145.9,183.1,119.3,180.3,168.5,231.8,224.5,192.8,122.9,336.5,185.9,
        194.3,149.5,210.1,273.3,191.4,287.0,226.0,303.6,289.9,421.6,264.5,342.3,
        339.7,440.4,315.9,439.3,401.3,437.4,575.5,407.6,682.0,475.3,581.3,646.9]
model = SARIMAX(data, order=(5,1,0))
res = model.fit(disp=False)
print("AIC = ", res.aic)

y = np.array(data)
model = SARIMAX(y, order=(5,1,0))
res = model.fit(disp=False)
print("AIC = ", res.aic)

y = pd.Series(data)
model = SARIMAX(y, order=(5,1,0))
res = model.fit(disp=False)
print("AIC = ", res.aic)
