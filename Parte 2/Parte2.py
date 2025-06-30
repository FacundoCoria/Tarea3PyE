import pandas as pd
import numpy as np
from scipy import stats

# Cargar los datos
df = pd.read_csv("velocidad_internet_ucu.csv")  

# Filtrar los edificios Central y Semprún
df_filtrado = df[df["Edificio"].isin(["Central", "Semprún"])]

# Separar las velocidades por edificio
central = df_filtrado[df_filtrado["Edificio"] == "Central"]["Velocidad Mb/s"]
semprun = df_filtrado[df_filtrado["Edificio"] == "Semprún"]["Velocidad Mb/s"]

# Calcular estadísticas
n1, n2 = len(central), len(semprun)
mean1, mean2 = central.mean(), semprun.mean()
std1, std2 = central.std(ddof=1), semprun.std(ddof=1)

# Calcular estadístico t manualmente
numerador = mean1 - mean2
denominador = np.sqrt((std1**2)/n1 + (std2**2)/n2)
t_stat = numerador / denominador

# Calcular grados de libertad con fórmula de Welch
s1_sq, s2_sq = std1**2, std2**2
df_numerador = ((s1_sq/n1) + (s2_sq/n2))**2
df_denominador = ((s1_sq/n1)**2)/(n1-1) + ((s2_sq/n2)**2)/(n2-1)
df_welch = df_numerador / df_denominador

# Calcular p-valor (unilateral inferior)
p_value = stats.t.cdf(t_stat, df=df_welch)

# Mostrar resultados
print("Media Central:", round(mean1, 2))
print("Media Semprún:", round(mean2, 2))
print("t:", round(t_stat, 4))
print("Grados de libertad:", round(df_welch, 2))
print("p-valor:", p_value)

if p_value < 0.05:
    print("Se rechaza H0: la velocidad en Central es significativamente menor.")
else:
    print("No se rechaza H0: no hay diferencia significativa.")
