import pandas as pd # Uso de dataframes para manejar tablas.
from scipy.stats import chi2 # Para calcular rango critico.
from scipy.stats import chisquare # Unicamente para verificar el resultado.

# Carga de datos
df = pd.read_csv('muestra_ech.csv')

# 1. Calcular ingreso per cápita
df['ingreso_per_capita'] = df['ingreso'] / df['personas_hogar']
print("=== Paso 1: Ingreso per cápita (primeros 5 hogares) ===")
print(df[['ingreso', 'personas_hogar', 'ingreso_per_capita']].head(), "\n") # Devuelve el ingreso per capita de los 5 primeros hogares
print("Cantidad de valores calculados:", df['ingreso_per_capita'].count()) # Verifica que se contaron todos los hogares.

# 2. Clasificar en quintiles según ingreso per cápita
df['quintil'] = pd.qcut(        # Crea los 5 quintiles.
    df['ingreso_per_capita'],
    q=5,
    labels=[1, 2, 3, 4, 5]
).astype(int)
print("=== Paso 2: Distribución en quintiles ===")
print(df['quintil'].value_counts().sort_index().to_frame(name='conteo'), "\n") # Cantidad de hogares por quintil.

# 3. Filtrar los hogares del quintil superior (quintil == 5)
quintil_superior = df[df['quintil'] == 5]
print("=== Paso 3: Hogares en el quintil superior ===")
print(quintil_superior[['departamento', 'ingreso_per_capita']], "\n")
print(f"Total hogares en quintil 5: {len(quintil_superior)}\n")

# 4. Tabla de frecuencias observadas de hogares ricos por departamento
obs = quintil_superior['departamento'].value_counts().sort_index()

print("=== Paso 4: Frecuencias observadas ===")
print(obs.to_frame(name='observadas'), "\n")

# 5. Frecuencias esperadas bajo hipótesis uniforme
todos_deptos = sorted(df['departamento'].unique())
obs = obs.reindex(todos_deptos, fill_value=0)
total_ricos = len(quintil_superior)
k = len(todos_deptos)
exp = pd.Series(total_ricos / k, index=todos_deptos)

print("=== Paso 5: Frecuencias esperadas ===")
print(exp.to_frame(name='esperadas'), "\n")

# 6. Estadístico chi-cuadrado
chi2_stat = ((obs - exp)**2 / exp).sum() # Forma manual

resultado = chisquare(f_obs=obs, f_exp=exp) # Forma con libreria de python. Solo utilizado para verificación.

# 7. Valor crítico para α=0.05 y df = k−1
alpha = 0.05
df_chi = k - 1
chi2_crit = chi2.ppf(1 - alpha, df_chi)

# 8. Conclusión
rechaza = chi2_stat > chi2_crit

print(f"Estadístico χ² = {chi2_stat:.2f}") # Resultado manual.
print(f"Cálculo scipy : {resultado.statistic:.2f}") # Unicamente para verificacion. Resultado con Scipy.

print(f"Valor crítico χ²({df_chi}, α={alpha}) = {chi2_crit:.2f}\n")

if rechaza:
    print("Se rechaza H₀: La distribución de hogares ricos NO ES uniforme entre departamentos.")
else:
    print("No se rechaza H₀: La distribución de hogares ricos ES uniforme entre departamentos.")
