import seaborn as sns
import os
import streamlit as st

# EDA pkgs
import pandas as pd

# Visualization pkgs
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def main():
    """Common ML Dataset Explorer"""
    st.title("Common ML Dataset Explorer")
    st.subheader("Simple Data Science Explorer with Streamlit")

    html_temp = """ 
    <div style="background-color:tomato;">
    <p>Haha</p>
    </div>
    """
    st.markdown(
        html_temp, unsafe_allow_html=True)  # important ti show the html in the app

    # load a csv file from the computer
    def file_selector(folder_path="./datasets"):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox("Select a file", filenames)
        return os.path.join(folder_path, selected_filename)

    filename = file_selector()
    st.info("You selected {}".format(filename))

    # Read Data
    df = pd.read_csv(filename)

    # Show Dataset
    if st.checkbox("Show Dataset"):
        number = st.number_input("Number of Rows to View", 5)
        st.dataframe(df.head(number))

    # Show Columns
    if st.button("Column Names"):
        st.write(df.columns)

    # Show Shape
    if st.checkbox("Shape of Dataset"):
        data_dim = st.radio("Show Dimension By", ("Rows", "Columns"))
        if data_dim == "Rows":
            st.text("Number of Rows")
            st.write(df.shape[0])
        elif data_dim == "Columns":
            st.text("Number of Columns")
            st.write(df.shape[1])
        else:
            st.write(df.shape)

    # Select Columns
    if st.checkbox("Select Columns To Show"):
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select", all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

    # Show Values
    if st.button("Value Counts"):
        st.text("Value Counts By Target/Class")
        st.write(df.iloc[:, -1].value_counts())

    # Show Datatypes
    # Show Values
    if st.button("Data Types"):
        st.write(df.dtypes)

    # Show Summary
    if st.checkbox("Summary"):
        st.write(df.describe().T)

    ## Plot and Visualization
    st.subheader("Data Visualization")

    # Correlation
    # Seaborn Plot
    if st.checkbox("Correlation Plot[Seaborn]"):
        st.write(sns.heatmap(df.corr(), annot=True))
        st.pyplot()

    # Count Plot
    if st.checkbox("Pie Plot"):
        all_columns_names = df.columns.tolist()
        if st.button("Generate Pie Plot"):
            st.success("Generating A Pie Plot")
            st.write(df.iloc[:, -1].value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()

    # Pie Chart
    # Customizable Plot

    st.subheader("Customizable Plot")
    all_columns_names = df.columns.tolist()
    type_of_plot = st.selectbox("Select Type of Plot", [
                                "area", "bar", "line", "hist", "box", "kde"])
    selected_columns_names = st.multiselect(
        "Select columns To Plot", all_columns_names)

    if st.button("Generate Plot"):
        st.success("Generating Customizable Plot of {} for {}".format(
            type_of_plot, selected_columns_names))

        # Plot by Streamlit [for area, bar and line]
        if type_of_plot == "area":
            custom_data = df[selected_columns_names]
            st.area_chart(custom_data)

        elif type_of_plot == "bar":
            custom_data = df[selected_columns_names]
            st.bar_chart(custom_data)

        elif type_of_plot == "line":
            custom_data = df[selected_columns_names]
            st.line_chart(custom_data)

        # Custom Plot [By Matplotlib or Seaborn for other types of charts...]
        elif type_of_plot:
            custom_plot = df[selected_columns_names].plot(kind=type_of_plot)
            st.write(custom_plot)
            st.pyplot()

    if st.button("Thanks"):
        st.balloons()

        st.sidebar.header("About App")
        st.sidebar.info("A Simple EDA App for Exploring Common ML Dataset")

        st.sidebar.header("Get Datasets")
        st.sidebar.markdown("[Common ML Dataset Repo]("")")

        st.sidebar.header("About")
        st.sidebar.info("Asmee Dhungana")
        st.sidebar.text("Built with Streamlit")
        st.sidebar.text("Tutorial: Jesse JCharis")


if __name__ == "__main__":
    main()
