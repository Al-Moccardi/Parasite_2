import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def load_data():
    data = pd.read_excel("temp_humid_data.xlsx", sheet_name='Sheet3')
    data['Date'] = pd.to_datetime(data['Date'])
    return data

# Funzione per addestrare il modello di regressione
def train_model(data):
    X = data[['temperature_mean', 'relativehumidity_mean']]
    y = data['no. of Adult males']
    pipeline = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100))
    pipeline.fit(X, y)
    return pipeline

def report_page():
    st.title('Report Dettagliato del Progetto')
    st.write("""
## Introduzione
Il progetto ha lo scopo di analizzare e prevedere il numero di parassiti adulti maschi basandosi su fattori ambientali quali temperatura e umidità, utilizzando dati storici per modellare le dinamiche temporali. Utilizzando tecniche di machine learning e visualizzazioni avanzate.

## Descrizione del Dataset
Il dataset utilizzato contiene le seguenti variabili principali:

1. Data: La data di osservazione, utilizzata per analizzare le tendenze temporali.
2. Temperatura Media: Media giornaliera della temperatura, espressa in gradi Celsius.
3. Umidità Relativa Media: Media giornaliera dell'umidità relativa, espressa in percentuale.
4. Numero di Adulti Maschi (Parassiti): Conteggio del numero di parassiti adulti maschi osservati. (target)
Il dataset copre un periodo da luglio a settembre 2023, offrendo una visione dettagliata dell'evoluzione delle condizioni ambientali e del loro impatto sui livelli di parassiti.

## Metodologia
La metodologia adottata include diverse fasi:

- Preparazione dei Dati: Pulizia e preparazione dei dati per l'analisi, inclusa la conversione delle date e la normalizzazione delle variabili numeriche.
- Visualizzazione dei Dati: Creazione di grafici interattivi per esplorare le relazioni tra temperatura, umidità e numero di parassiti.
- Modello di Machine Learning: Addestramento di un modello di regressione (RandomForestRegressor) per prevedere il numero di parassiti basandosi su temperatura e umidità.
- Valutazione del Modello: Analisi dell'importanza delle variabili e valutazione delle performance del modello.
Analisi e Visualizzazioni
- Andamento Temporale: Il grafico mostra come variano umidità e numero di parassiti nel tempo, codificando la temperatura tramite il colore dei punti. Questo aiuta a comprendere come le condizioni ambientali influenzano la presenza dei parassiti.
- Relazione 3D: Un grafico tridimensionale che mette in relazione temperatura, umidità e numero di parassiti, fornendo una visualizzazione complessa delle interazioni tra queste variabili.
- Relazioni Bivariate: Grafici che esplorano la relazione diretta tra temperatura e umidità con il numero di parassiti, utilizzando la grandezza dei punti per rappresentare il numero di parassiti.
- Distribuzioni Multivariate: Analisi della distribuzione congiunta di temperatura e umidità, con un focus sui valori target specifici.
    """)
def main_page():
    st.title("Dashboard di Analisi e Predizione dei Parassiti")
    data = load_data()
    model = train_model(data)

    # Sidebar per filtri e controlli
    st.sidebar.subheader("Filtri e Controlli di Visualizzazione")
    start_date = st.sidebar.date_input("Data Iniziale", data['Date'].min().date())
    end_date = st.sidebar.date_input("Data Finale", data['Date'].max().date())

    # Filtra i dati in base all'intervallo di date selezionato
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

    # Scatterplot temporale
    fig_temporal_scatter = px.scatter(filtered_data, x='Date', y='relativehumidity_mean',
                                      size='no. of Adult males', color='temperature_mean',
                                      labels={'no. of Adult males': 'Numero di Adulti Maschi', 'temperature_mean': 'Temperatura Media'},
                                      title="Andamento Temporale: Umidità vs Numero di Adulti Maschi")
    st.plotly_chart(fig_temporal_scatter, use_container_width=True)
    
    st.write(""" l'andamento presenta un trend crescente del numero dei parassiti codificati in base alla grandezza dello scatter variando in maniera più significativa sulla base temporale  (Settembre) che presenta combinazioni di valori di temperatura e umidità che favoriscono la crescita dei parassiti""")
    # Scatterplot 3D
    fig_scatter_3d = px.scatter_3d(filtered_data, x='temperature_mean', y='relativehumidity_mean', z='no. of Adult males',
                                   color='no. of Adult males', labels={"no. of Adult males": "Numero di Adulti Maschi"},
                                   title="Relazione 3D tra Temperature, Umidità e Adulti Maschi")
    st.plotly_chart(fig_scatter_3d, use_container_width=True)
    st.write("""Il grafico 3D mostra come la temperatura e l'umidità influenzino il numero di parassiti adulti maschi, con una maggiore concentrazione di punti in alcune regioni dello spazio tridimensionale. Questo suggerisce che esistono combinazioni specifiche di temperatura e umidità che favoriscono la crescita dei parassiti.""")
    # Scatterplot codificato
    fig_scatter = px.scatter(filtered_data, x='temperature_mean', y='relativehumidity_mean',
                             size='no. of Adult males', color='no. of Adult males',
                             title="Relazione tra Temperatura, Umidità e Numero di Adulti Maschi")
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.write(""" La densità e temperatura cominciano ad emergere come parametri importante per determinare il un numero di parassiti""")
    # Contour plot
    fig_contour = px.density_contour(filtered_data, x='temperature_mean', y='relativehumidity_mean', 
                                     marginal_x='histogram', marginal_y='histogram',
                                     title="Distribuzioni Multivariate con Valori Target")
    fig_contour.add_trace(px.scatter(filtered_data, x='temperature_mean', y='relativehumidity_mean', 
                                     size='no. of Adult males').data[0])
    st.plotly_chart(fig_contour, use_container_width=True)
    st.write("""E possibile notare come la temperatura possiede meno variazione rispetto all'umidità e come la densità di parassiti aumenti con l'aumento dell'umidità o a temperature più basse ed umidità normale""")

    # Sezione di previsione
    st.sidebar.subheader("Predizione di Adulti Maschi")
    temp_input = st.sidebar.number_input("Inserisci la Temperatura Media", value=float(data['temperature_mean'].mean()))
    humidity_input = st.sidebar.number_input("Inserisci l'Umidità Relativa Media", value=float(data['relativehumidity_mean'].mean()))
    if st.sidebar.button("Prevedi"):
        prediction = model.predict([[temp_input, humidity_input]])[0]
        prediction = max(0, int(round(prediction)))  # Arrotonda e imposta minimo a 0
        st.sidebar.write(f"Numero previsto di adulti maschi: {prediction}")
        # Istogramma della distribuzione della variabile target con valore predetto
        fig = px.histogram(filtered_data, x='no. of Adult males', nbins=30, title="Distribuzione di Numero di Adulti Maschi",
                           marginal="violin", hover_data=filtered_data.columns)
        fig.add_vline(x=prediction, line_width=3, line_dash="dash", line_color="red")
        fig.update_layout(bargap=0.2)
        st.plotly_chart(fig, use_container_width=True)
        st.write("""Il valore predetto è rappresentato dalla linea rossa, che indica la posizione stimata del numero di parassiti adulti maschi sulla distribuzione. Questo fornisce un'indicazione visiva del confronto tra il valore previsto e la distribuzione osservata.""")
        st.write(""" provare il modello ad una serie di valori di temperatura e umidità per vedere come varia il numero di parassiti e validare le analisi svolte""")
def main():
    st.sidebar.title("Navigazione")
    page = st.sidebar.radio("Seleziona la pagina:", ['Pagina Principale', 'Report Dettagliato'])

    if page == 'Pagina Principale':
        main_page()
    elif page == 'Report Dettagliato':
        report_page()
        
if __name__ == "__main__":
    main()
