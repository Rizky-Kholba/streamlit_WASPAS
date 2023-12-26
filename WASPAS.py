# Author: Muhamad Rizky Kholba

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class WASPASDecision:
    def __init__(self, data, bobot, tipeKriteria, namaAlternatif, namaKriteria):
        self.data = data
        self.bobot = bobot
        self.tipeKriteria = tipeKriteria
        self.namaAlternatif = namaAlternatif
        self.namaKriteria = namaKriteria
        self.dataTernormalisasi = self.normalisasiData()
        self.nilaiQi = self.hitung_nilaiQi()

    def normalisasiData(self):
        dataTernormalisasi = np.zeros_like(self.data, dtype=np.float64)

        for j in range(self.data.shape[1]):
            if self.tipeKriteria[j] == "benefit":
                max_value = np.max(self.data[:, j])
                dataTernormalisasi[:, j] = self.data[:, j] / max_value
            elif self.tipeKriteria[j] == "cost":
                min_value = np.min(self.data[:, j])
                dataTernormalisasi[:, j] = min_value / self.data[:, j]

        return dataTernormalisasi

    def hitung_nilaiQi(self):
        nilaiQi = np.zeros(self.dataTernormalisasi.shape[0])

        for i in range(self.dataTernormalisasi.shape[0]):
            sum_term = (0.5) * (np.sum(self.dataTernormalisasi[i, :] * self.bobot))
            prod_term = (0.5) * (np.prod(self.dataTernormalisasi[i, :] ** self.bobot))

            nilaiQi[i] = sum_term + prod_term

        return nilaiQi

    def tampilanOutput(self):
        # Menampilkan label C1, C2, dll., untuk kolom
        column_labels = [f"C{i+1}" for i in range(self.data.shape[1])]
        # Menampilkan label A1, A2, dll., untuk baris
        row_labels = [f"A{i+1}" for i in range(self.data.shape[0])]

        # Menampilkan grafik melingkar untuk persentase bobot kriteria
        st.subheader("\nGrafik Persentase Bobot Kriteria")

        # explode = [0.02 if t == "benefit" else 0.02 for t in self.tipeKriteria]
        explode = [0.02] * len(self.tipeKriteria)
        colors = [
            "lightgreen" if t == "benefit" else "lightcoral" for t in self.tipeKriteria
        ]

        fig, ax = plt.subplots()
        ax.pie(
            self.bobot,
            labels=self.namaKriteria,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors,
            wedgeprops=dict(width=0.55, edgecolor="gray"),
            explode=explode,
            # shadow={"ox": -0.04, "edgecolor": "none", "shade": 0.9},
        )
        ax.axis("equal")
        st.pyplot(fig)

        # Menampilkan grafik nilai kriteria setiap alternatif
        st.subheader("\nGrafik Nilai Kriteria Setiap Alternatif")

        # Transpose data untuk memudahkan plotting
        data_transposed = self.data.T
        num_attributes = data_transposed.shape[0]
        num_alternatives = data_transposed.shape[1]

        x = np.arange(num_alternatives)  # the label locations
        width = 0.2  # the width of the bars
        multiplier = 0

        fig, ax = plt.subplots()
        max_value = np.max(self.data)  # Ambil nilai maksimal dari seluruh data

        for i, (attribute, measurement) in enumerate(
            zip(self.namaKriteria, data_transposed)
        ):
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(
                rects, padding=3, labels=measurement
            )  # Menggunakan nilai data sebagai label
            multiplier += 1

        # Set batas nilai y sesuai dengan nilai maksimal
        ax.set_ylim(0, max_value + 2)  # Sesuaikan batas nilai sesuai kebutuhan

        # Add some text for labels, title, and custom x-axis tick labels, etc.
        ax.set_ylabel("Nilai Kriteria")
        ax.set_title("Nilai Kriteria Setiap Alternatif")
        ax.set_xticks(x + (width * num_attributes) / 2)
        ax.set_xticklabels(self.namaAlternatif)
        ax.legend(loc="upper left", ncol=num_attributes)

        st.pyplot(fig)

        # Menampilkan matriks keputusan dengan label yang sudah dimodifikasi
        st.subheader("\nMatriks Keputusan:")
        st.table(pd.DataFrame(self.data, columns=column_labels, index=row_labels))

        # Menampilkan bobot
        st.subheader("\nBobot Kriteria:")
        bobot_df = pd.DataFrame(
            {
                "Persentase": [f"{weight * 100:.0f}%" for weight in self.bobot],
                "Desimal": self.bobot,
            },
            index=column_labels,
        )
        st.table(bobot_df)

        # Menampilkan matriks ternormalisasi dengan label yang sudah dimodifikasi
        st.subheader("\nMatriks Ternormalisasi:")
        st.table(
            pd.DataFrame(
                self.dataTernormalisasi, columns=column_labels, index=row_labels
            )
        )

        # Menampilkan nilai aletrnatif terbaik
        best_alternative_index = np.argmax(self.nilaiQi) + 1

        st.subheader("\nNilai Qi:")
        for i, value in enumerate(self.nilaiQi, start=1):
            st.write(f"(Q{i}) {self.namaAlternatif[i-1]}: {value}")

        # Menampilkan grafik nilai Qi untuk setiap alternatif
        st.subheader("\nGrafik Nilai Qi Setiap Alternatif:")

        fig, ax = plt.subplots()
        x_qi = np.arange(len(self.namaAlternatif))
        qi_values = self.nilaiQi

        ax.bar(x_qi, qi_values, color="skyblue")
        ax.set_xticks(x_qi)
        ax.set_xticklabels(self.namaAlternatif)
        ax.set_ylabel("Nilai Qi")
        ax.set_title("Nilai Qi Setiap Alternatif")

        st.pyplot(fig)

        # Kesimpulan
        st.subheader("\nKesimpulan:")
        st.markdown(
            f"\nNilai :red[Q{best_alternative_index}] memiliki nilai paling besar, sehingga :red[(A{best_alternative_index}) {self.namaAlternatif[best_alternative_index-1]}] terpilih sebagai alternatif terbaik."
        )


# Aplikasi Streamlit
def main():
    st.title(":bar_chart: Sistem Pengabil Keputusan Metode WASPAS\n")

    # Input jumlah kriteria dan alternatif
    st.subheader("Jumlah Kriteria dan Alternatif")

    num_criteria = st.number_input("Jumlah Kriteria", min_value=1, step=1)
    num_alternatives = st.number_input("Jumlah Alternatif", min_value=1, step=1)

    # Pendefinisian kriteria
    st.subheader("Pendefinisian Kriteria")

    namaKriteria = []
    tipeKriteria = []
    bobot = []

    col1, col2, col3 = st.columns(3)

    with col1:
        # Input nama kriteria
        for i in range(num_criteria):
            namaKriteria.append(st.text_input(f"Nama Kriteria {i+1}"))

    with col2:
        # Input bobot kriteria sebagai persentase
        for i in range(num_criteria):
            tipeKriteria.append(
                st.selectbox(
                    f"Jenis Kriteria {namaKriteria[i]}",
                    ["benefit", "cost"],
                    key=f"selectbox_{i}",
                )
            )

    with col3:
        # Input kriteria dan jenis kriteria (benefit atau cost)
        for i in range(num_criteria):
            bobot.append(
                st.number_input(
                    f"Bobot Kriteria {namaKriteria[i]} (%)",
                    min_value=0,
                    max_value=100,
                    step=1,
                    key=f"weight_{i}",
                )
                / 100.0
            )

    # Validasi total bobot kriteria
    if sum(bobot) != 1.0:
        st.error("Total bobot kriteria harus sama dengan 100%.")
        return

    # Pendefinisian alternatif
    st.subheader("Pendefinisian Alternatif")

    # Input nama alternatif
    namaAlternatif = []
    for i in range(num_alternatives):
        namaAlternatif.append(st.text_input(f"Nama Alternatif {i+1}"))

    # Nilai matriks
    st.subheader("Nilai Matriks")

    # Input matriks keputusan
    data = np.zeros((num_alternatives, num_criteria))
    for i in range(num_alternatives):
        for j in range(num_criteria):
            data[i, j] = st.number_input(
                f"(A{i+1}) Alternatif {namaAlternatif[i]} - (C{j+1}) Kriteria {namaKriteria[j]}",
                min_value=0,
                step=1,
                key=f"input_{i}_{j}",
            )

    # Membuat objek WASPASDecision
    waspas_model = WASPASDecision(
        data, np.array(bobot), tipeKriteria, namaAlternatif, namaKriteria
    )

    # Tombol Hitung
    if st.button("Hitung"):
        waspas_model.tampilanOutput()


if __name__ == "__main__":
    main()
