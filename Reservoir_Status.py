import sqlite3
import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from pandas.plotting import register_matplotlib_converters
import os
import pandas as pd
import pytz
from datetime import datetime, timedelta
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import platform

# Purpose:
#       Create a graph showing a combination of reservoir storage and snow-pack in terms to total acre feet.
# Methods:
#       - Obtain reservoir storage levels through a PI web interface.
#       - Obtain snow-pack information through SNODAS interface (this program lives in the GIS-grib directory).
#       - Plot the Reservoir storage, snow-pack storage, and combined storage levels on plot.
# Params:
#       - Although nothing is passed, this program relies upon the raster_builder_snodas.py program in the GIS-grib
#         directory.


def main():
    # Process of Main:
    #       1) Make PI request to get current storage levels for French Meadows and Hell Hole
    #           a) store PIU request in dataframe with datetime as index
    #       2) Combine both dataframes into a single dataframe, merged off of Date index
    #       3) Get snowdas data and place into dataframe
    #           TODO: Move SNODAS data from excel format to SQL
    #       4) Merge snowdas dataframe into storage dataframe, merge off of month, day from storage df
    #       5) Send df to plot.

    today = datetime.today()                                                # Find the water year.
    wy = today.year
    this_dir = os.path.dirname(os.path.realpath(__file__))
    if today.month >= 10:                                                   # Advance WY if current month is Oct-Dec.
        wy = today.year + 1
    df_FM = reservoir_request("French Meadows", wy)                         # Create df from French Meadows PI data.
    df_HH = reservoir_request("Hell Hole", wy)                              # Create df from Hell Hole PI data.

    # Merge the df's off of month and day, only keep Value columns.
    # Reset index to the FM date (if you
    # don't reset the index, it will default to a
    # numbered list of n1, n2,...n)
    df_final = pd.merge(df_FM.filter(like='Value', axis='columns'), df_HH,
                        left_on=[df_FM.index.month, df_FM.index.day],
                        right_on=[df_HH.index.month, df_HH.index.day],
                        how='left').set_index(df_FM.index)

    df_final.drop(['key_0', 'key_1'], inplace=True, axis=1)                 # key values not needed.

    # We are using a excel file to store the normal values form both
    # the reservoir data and the snowdas data. This is used on the
    # graph to display the % of normal.
    historical_data = os.path.join(this_dir,'MFP_Historical_Reservoir_Data.xlsx')
    dfh = pd.read_excel(historical_data, sheet_name='Data')

    dfh.Date = pd.to_datetime(dfh.Date)

    dfh.set_index(pd.DatetimeIndex(dfh.Date), inplace=True)

    # Put the historical data into our main dataframe (easier to plot this way).
    df_final = pd.merge(df_final, dfh,
                        left_on=[df_final.index.month, df_final.index.day],
                        right_on=[dfh.index.month, dfh.index.day],
                        how='left').set_index(df_final.index)

    df_final.drop(['key_0', 'key_1'], inplace=True, axis=1)

    # A change greater than 50,0000 AF in one day is an error, so remove the bad data.
    df_final['Total_Storage'] = df_final['Value_x'] + df_final['Value_y']
    df_final = df_final[df_final['Total_Storage'].shift(-1) - df_final['Total_Storage'] < 50000]

    df_snow = snowpack()

    # Remove the nan values before the merge. If we don't do this the naT values
    # within the datetime index will create problems during the merge.
    df_snow = df_snow[df_snow.index.notna()]

    df_final = pd.merge(df_final, df_snow['Snowpack'],
                        left_on=[df_final.index.year, df_final.index.month, df_final.index.day],
                        right_on=[df_snow.index.year, df_snow.index.month, df_snow.index.day],
                        how='left').set_index(df_final.index)

    simple_plot(df_final, dfh, wy)
    print("Success! Image Saved")
    # write entire Excel file from the Energy Marketing folder to sql table
    sql_create_table(df_snow)
    return

def flow_request(wy):
    # Process:
    #   1) First request: Access the PI interface to get the weblink
    #       -This is probably just an extra step at this point, but if we just used the current web URL for a station,
    #        it's possible it could change. Therefore, I felt it was safer to simply search for the URL first,
    #        then get the information using the url.
    #   2) Second request: now that we have the weblink, download the PI data.
    #
    # Note: The following code accesses PCWA's PI database. What follows directly below in the comments are
    #       a bunch of notes on how to sift through the data. Additional info can be found in OneNote under
    #       the PI+Python page.
    #
    # To find information through the URL start here: https://flows.pcwa.net/piwebapi/elements
    # For reservoirs, it will look like this:
    # https://flows.pcwa.net/piwebapi/elements/?path=\\BUSINESSPI2\OPS\Reservoirs
    # Click "Elements"--> Reservoir Name --> Attributes.

    # For gauges it will look like this:
    #  https://flows.pcwa.net/piwebapi/elements/?path=\\BUSINESSPI2\OPS\Gauging%20Stations
    # 		○ Click "Elements" --> Gauge Name --> "Attributes"
    # The full path then looks something like this:
    # https://flows.pcwa.net/piwebapi/attributes/?path=\\BUSINESSPI2\OPS\Gauging%20Stations\R30|Flow

    # Notice above that the parameter "Path" is:
    # \\\\BUSINESSPI2\\OPS\\Gauging Stations\\R30|Flow
    #
    # If you put that into your requests path above if would not work (i.e.
    # requests.get( url="https://flows.pcwa.net/piwebapi/elements",
    # params={"path": "\\\\BUSINESSPI2\\OPS\\Gauging Stations\\R30|Flow", } )
    #
    # When you want anything after a pipe (i.e. a "|") you have to use a different path
    # (notice "/elements" is replaced with /attributes") : https://flows.pcwa.net/piwebapi/attributes/

    # This first request simply obtains the weblink. This may be overkill, but I still think this is the
    # safer way to go.
    try:
        response = requests.get(
            url="https://flows.pcwa.net/piwebapi/attributes",
            params={"path": "\\\\BUSINESSPI2\\OPS\\Gauging Stations\\R30|Flow",
                    },
        )

        # Convert response to json so we can put it in a dataframe
        j = response.json()
        url_flow = j['Links']['InterpolatedData']
    except requests.exceptions.RequestException:
        print('HTTP Request failed')

    # Now that we have the url for the PI data, this request is for the actual data. We will
    # download data from the beginning of the water year to the current date. (We can't download data
    # past today's date, if we do we'll get an error.
    try:
        response = requests.get(
            url= url_flow ,
            params={"startTime": f"{wy-1}-10-01T00:00:00-07:00",
                    "endTime": datetime.today().strftime("%Y-%m-%dT00:00:00-07:00"),
                    "interval": "1d",
                    },
        )
        # print('Response HTTP Status Code: {status_code}'.format(status_code=response.status_code))
        # print('Response HTTP Response Body: {content}'.format(content=response.content))
        j = response.json()
        df = pd.DataFrame.from_dict((j["Items"]))
        df.index = pd.to_datetime(df.Timestamp)
        df.index = df.index.tz_convert('US/Pacific')
        create_plot(df)
    except requests.exceptions.RequestException:
        print('HTTP Request failed. Is PI data for ')

def reservoir_request(resName,wy):
    # Process:
    #   1) First request: Access the PI interface to get the weblink
    #       -This is probably just an extra step at this point, but if we just used the current web URL for a station,
    #        it's possible it could change. Therefore, I felt it was safer to simply search for the URL first,
    #        then get the information using the url.
    #   2) Second request: now that we have the weblink, download the PI data.
    #
    # @Params:
    #   resName: name of the reservoir --> French Meadows or Hell Hole.
    #   wy:      Current water year.
    #
    # Note: The following code accesses PCWA's PI database. What follows directly below in the comments are
    #       a bunch of notes on how to sift through the data. Additional info can be found in OneNote under
    #       the PI+Python page.
    #
    # To find information through the URL start here: https://flows.pcwa.net/piwebapi/
    # Then Click on "AssetServers" --> Database Name --> "Databases" --> Group (i.e. "OPS") --> "Elements"
    # For reservoirs, it will look like this:
    # https://flows.pcwa.net/piwebapi/elements/?path=\\BUSINESSPI2\OPS\Reservoirs
    # Click "Elements"--> Reservoir Name --> Attributes.

    # For gauges it will look like this:
    #  https://flows.pcwa.net/piwebapi/elements/?path=\\BUSINESSPI2\OPS\Gauging%20Stations
    # 		○ Click "Elements" --> Gauge Name --> "Attributes"
    # The full path then looks something like this:
    # https://flows.pcwa.net/piwebapi/attributes/?path=\\BUSINESSPI2\OPS\Gauging%20Stations\R30|Flow

    # Notice above that the parameter "Path" is:
    # \\\\BUSINESSPI2\\OPS\\Gauging Stations\\R30|Flow
    #
    # If you put that into your requests path above if would not work (i.e.
    # requests.get( url="https://flows.pcwa.net/piwebapi/elements",
    # params={"path": "\\\\BUSINESSPI2\\OPS\\Gauging Stations\\R30|Flow", } )
    #
    # When you want anything after a pipe (i.e. a "|") you have to use a different path
    # (notice "/elements" is replaced with /attributes") : https://flows.pcwa.net/piwebapi/attributes/

    # This first request simply obtains the weblink. This may be overkill, but I still think this is the
    # safer way to go.
    try:
        response = requests.get(
            url="https://flows.pcwa.net/piwebapi/attributes",
            params={"path": f"\\\\BUSINESSPI2\\OPS\\Reservoirs\\{resName}|Storage",
                    },
        )

        j = response.json()
        url_flow = j['Links']['InterpolatedData']
    except requests.exceptions.RequestException:
        print('HTTP Request failed')

    # Now that we have the url for the PI data, this request is for the actual data. We will
    # download data from the beginning of the water year to the current date. (We can't download data
    # past today's date, if we do we'll get an error.
    try:
        response = requests.get(
            url=url_flow,
            params={"startTime": f"{wy - 1}-10-01T00:00:00-07:00",
                    "endTime": datetime.today().strftime("%Y-%m-%dT00:00:00-07:00"),
                    "interval": "1d",
                    },
        )
        print('Response HTTP Status Code: {status_code}'.format(status_code=response.status_code))

        # Place response into json so that it can then be put into dataframe.
        j = response.json()

        # We only want the "Items" object.
        df = pd.DataFrame.from_dict((j["Items"]))

        # Convert the Timestamp to a pandas datetime object and convert to Pacific time.
        df.index = pd.to_datetime(df.Timestamp)
        df.index = df.index.tz_convert('US/Pacific')

        # This section is needed because the French Meadows gauge is broken. The manual readings are
        # placed into an Excel file, which the following code will open and read. The code will
        # the dataframe with the PI data and replace it with any data from the manual readings.
        if resName == 'Put Res Name Here':
            FMfile = "G:/Energy Marketing/Trackers/FM WSE Manual Read (April 2020).xlsx"
            df_manual = pd.read_excel(FMfile, sheet_name='Sheet1', header=1)

            # Reset the index to the date
            df_manual.index = pd.to_datetime(df_manual.Date)

            # Rename all the columns
            df_manual.columns = ['Date','Time','Elevation','Value']

            # Merge the two dataframes based on the PI data dataframe.
            df = pd.merge(df.filter(like='Value', axis='columns'), df_manual,
                                left_on=[df.index.month, df.index.day],
                                right_on=[df_manual.index.month, df_manual.index.day],
                                how='left').set_index(df.index)

            #  "Value_x" is the original "Value" column, "Value_y" is the manual entry data. This says: The new Value
            # column will be all the data from the Manual entry, except where there's an nan (which will be any
            # date that's not in the manual file), in that case it will be the PI data.
            df['Value'] = df['Value_y'].fillna(df['Value_x'])

            # Get rid of any useless data.
            df.drop(['key_0', 'key_1','Value_x','Value_y','Date','Time','Elevation'], axis=1, inplace=True)

        return df
    except requests.exceptions.RequestException:
        print('HTTP Request failed')

def snowpack():
    # Purpose:
    #   - Open the SNODAS Excel sheet and put the entire spreadsheet into a dataframe.
    # Methods:
    #   - This code is extremely basic and relies upon successful completion of the GIS_Grib program. The
    #     GIS_Grib program will put the SNODAS data into the Daily_Output.xlsx file. The
    #     current program will open that file and ingest two sheets. One sheet for Hell Hole and the other for
    #     French Meadows. Each sheet will be placed into a Dataframe, then combined into a single dataframe.
    try:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        # Create a dataframe out of the French Meadows spreadsheet.
        file = os.path.join(os.path.sep,'home','smotley','programs','data','Daily_Output.xlsx')
        if platform.system() == 'Windows':
            file = os.path.join('G:/Energy Marketing/Weather', 'Daily_Output.xlsx')
        df1 = pd.read_excel(file, sheet_name='French_Meadows')
        df1.Date = pd.to_datetime(df1.Date)
        df1.set_index(pd.DatetimeIndex(df1.Date), inplace=True)

        # Create a dataframe out of the Hell Hole spreadsheet.
        df2 = pd.read_excel(file, sheet_name='Hell_Hole')
        df2.Date = pd.to_datetime(df2.Date)
        df2.set_index(pd.DatetimeIndex(df2.Date), inplace=True)

        # Merge sheets on the index (date).
        df_final = pd.merge(df1, df2, left_on=df1.index,right_on=df2.index, how='left')
        df_final.set_index(df_final['Date_x'], inplace=True)

        # The total snowpack will be the combination of each basin.
        df_final['Snowpack'] = df_final['TotalAF'] + df_final['TotalAF_HellHole']
        return df_final

    except Exception as err:
        print("Reading Daily_Output.xlsx Failed. Error: " + str(err))
        combined_basin_tot = -99999999
        combined_change = -99999999
        return None



def simple_plot(df, dfh, wy):
    # Purpose:
    #       - Create a plot using two dataframes:
    #           - 1) Dataframe containing reservoir and snowpack info.
    #           - 2) Dataframe containing historical snowpack and reservoir information.

    fig, ax1 = plt.subplots()

    # The last date in the dataframe (if everything is updated, this should be yesterday's date)
    as_of_date = (df['Date'].iloc[-1]).strftime("%#m/%#d/%Y")

    # The main title
    plt.suptitle('MFP Reservoir and Snowpack Storage', fontsize=12, ha='center')

    # The subtitle
    plt.title(f'As of {as_of_date}', fontsize=8, ha='center')

    # x axis date format as short month name.
    date_format = mdates.DateFormatter("%b")
    ax1.xaxis.set_major_formatter(date_format)

    ax1.set_ylabel('Acre Feet (AF)', color='black', fontsize=8)
    ax1.set_xlabel(f'Water Year {wy}', fontsize=8)

    # Put a comma after the thousands place (e.g. 1,000 instead of 1000)
    ax1.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    # ax1.yaxis.grid(True) # Turn on grid

    # --Start-- Create a color gradient under the curve
    alpha = 1.0
    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb('tab:blue')
    z[:, :, :3] = rgb
    z[:, :, -1] = np.linspace(0, alpha, 100)[:, None]

    # The extent of the gradient will be the first and last dates in the df (xaxis) and 0 to 320000.
    # If you want to flip the gradient, this is where to do it.
    xmin, xmax, ymin, ymax = mdates.date2num(df.index.values).min(), \
                             mdates.date2num(df.index.values).max(), 0, 320000

    # Plot the gradient
    im = ax1.imshow(z, aspect='auto', extent=[xmin, mdates.datestr2num(f'{wy}-10-01'), ymin, ymax],
                    origin='lower', zorder=1)

    # This allows for a continuous line in the snowpack graph (even if nan).
    mask = np.isfinite(df['Snowpack'])

    # The xy values of the gradient will now be a continuous line (due to the masK)  with the x values
    # being dates and the y values being a combination of the snowpack and total storage.
    xy = np.column_stack([mdates.date2num(df.index.values[mask]),  df['Snowpack'][mask]+df['Total_Storage'][mask]])

    #xy = np.column_stack([mdates.date2num(df.index.values), df['Total_Storage']])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])

    # Make the gradient polygon shape
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax1.add_patch(clip_path)
    im.set_clip_path(clip_path)
    ax1.plot(df.index.values[mask],  df['Snowpack'][mask]+df['Total_Storage'][mask], color='royalblue', alpha=0.7)
    # --End-- Color Gradient

    # Add a constant value on the chart of the storage capacity as a dashed line.
    ax1.plot(dfh['Storage Capacity'], linestyle='dashed')

    # Add a line plot of the total storage as a blue line and fill the area below that line with 'cornflowerblue'
    ax1.plot(df.index, df['Total_Storage'], color='navy', alpha=0.7)
    ax1.fill_between(df.index, df['Total_Storage'], color='cornflowerblue', alpha=0.7)

    # Add the snowpack line as a white line.
    ax1.plot(df.index[mask], df['Snowpack'][mask], color='white', alpha=0.7)

    # Get the last value in the graph for total storage, snowpack, and combined.
    storage_today = int(df['Total_Storage'].iloc[-1])
    snowpack_today = int(df['Snowpack'].iloc[-1])
    combined_storage = storage_today + snowpack_today

    # Find the value in the historical data that matches the last date in our data (the most recent data).
    # This will create a mask where all values in the pcnt_mask = False except for the date that matches.
    pcnt_mask = (dfh['Date'] == df['Date'].iloc[-1])

    # Find the normal reservoir value and snowpack value in the historical database for today's date.
    pcnt_normal_res = int((df['Total_Storage'].iloc[-1] / dfh['Historical Average'].loc[pcnt_mask].values[-1]) * 100)
    pcnt_normal_snow = int((snowpack_today / dfh['Historical Snowpack'].loc[pcnt_mask].values[-1]) * 100)

    #combined_storage = int(df['Snowpack'].iloc[-1]+df['Total_Storage'].iloc[-1])
    xytext_x = 15
    xytext_y = 0

    # By the time we get to June, we need to shift the annotations over.
    maxDate = df['Date'].iloc[-1]
    #if maxDate.month > 5 and maxDate.month < 10:
    #    xytext_x = -100

    # Position of the "Reservoir" storage level text on the graph. Move label down if overlaps "combined storage" text
    xytext_res = (xytext_x, xytext_y)
    if storage_today + 30000 > combined_storage:
        xytext_res = (xytext_x, -20)

    # Position the "Snowpack" label higher in the y-direction if the snowpack is getting low
    xytext_snowpack = (xytext_x, xytext_y)
    if snowpack_today < 10000:
        xytext_snowpack = (xytext_x, 10)

    # If the reservoir and snowpack are ~same (in acre feet) they will overlap, so move the snowpack text down (y-dir)
    if abs(snowpack_today - storage_today) < 25000:
        xytext_snowpack = (xytext_x, -20)

    # Place Label for Reservoir Storage.
    ax1.annotate(f'{storage_today:,} Reservoir  \n ({pcnt_normal_res}% of Average)',
                 (xmax, storage_today), xytext=xytext_res, fontsize=8,
                 textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))

    # Place label for snowpack if there's at least 10 AF of snowpack
    if snowpack_today > 10:
        ax1.annotate(f'{snowpack_today:,} Snowpack  \n ({pcnt_normal_snow}% of Average)',
                     (xmax, snowpack_today), xytext=xytext_snowpack, fontsize=8,
                     textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))

    # If there's still snow on the ground, include the label for the combined storage.
    if snowpack_today > 100:
        ax1.annotate(f'{combined_storage:,} \n Combined Storage \n (Reservoir + Snowpack)',
                     (xmax, combined_storage), xytext=(15, 0), fontsize=8, weight='bold',
                     textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))

    ax1.annotate('Capacity = 342,583', (mdates.date2num(dfh.index.values[-1])-85, 347000), xytext=(15, 0),
                 textcoords='offset points', fontsize=6, zorder=8)

    ax1.annotate('SNOWPACK', (mdates.date2num(dfh.index.values[-1])-275, 30000), xytext=(15, 0),
                 weight='bold', color='white', textcoords='offset points', fontsize=9, zorder=9)

    ax1.annotate('RESERVOIR', (mdates.date2num(dfh.index.values[-1])-350, 100000), xytext=(15, 0),
                 weight='bold', color='navy', textcoords='offset points', fontsize=9, zorder=10)

    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    imgdir = os.path.join(os.path.sep,'home','smotley','images','weather_email')
    if platform.system() == "Windows":
        imgdir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Images'))
    plt.savefig(os.path.join(imgdir, 'MFP_Combined_Storage.png'))
    return

def create_plot(df):
    fig, ax1 = plt.subplots()
    plt.title('Precipitation and Snow Level Forecast: Middle Fork')

    color = 'tab:blue'
    date_format = mdates.DateFormatter("%a %m/%d")
    ax1.xaxis.set_major_formatter(date_format)

    xaxis_lowlimit = datetime.now(pytz.timezone('US/Pacific')) - timedelta(days=40)
    xaxis_uplimit = datetime.now(pytz.timezone('US/Pacific'))
    ax1.set_xlim([xaxis_lowlimit, xaxis_uplimit])

    ax1.set_ylabel('Snow Level', color=color)
    ax1.set_xlabel('Date')
    ax1.set_ylim([0.0, 10000])

    #ax1.yaxis.grid(True)

    # --Start-- Create a color gradient under the curve
    alpha = 1.0
    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(color)
    z[:, :, :3] = rgb
    z[:, :, -1] = np.linspace(0, alpha, 100)[:, None]

    xmin, xmax, ymin, ymax = mdates.date2num(df.index.values).min(), \
                             mdates.date2num(df.index.values).max(), 0, 2000

    im = ax1.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=1)

    xy = np.column_stack([mdates.date2num(df.index.values), df['Value']])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax1.add_patch(clip_path)
    im.set_clip_path(clip_path)
    # --End-- Color Gradient

    ax2 = ax1.twinx()
    #ax2.set_ylim([0.0, 3.0])
    ax2.xaxis.set_major_formatter(date_format)
    color = 'tab:green'
    ax2.set_ylabel('QPF', color=color)
    ax2.bar(df.index, df[f'Value'], color=color, alpha=0.7)
    ax2.bar(df.index, df[f'Value'], color='tab:blue', alpha=0.7)

    rects = ax2.patches
    # Make some labels.
    (y_bottom, y_top) = ax2.get_ylim()
    y_height = y_top - y_bottom

    for rect_cnt, rect in enumerate(rects):
        height = rect.get_height()
        label = height
        # If a precip value is off the chart, put the label at the top of the bar so the value is known.
        if height > y_top:
            height = y_top - 0.06

        label_position = height + (y_height * 0.01)

        # Since the 06Z run will include forecast times valid yesterday, we want to remove those from the graph
        # if we don't include this if statement first, it will plot values along the -side of the x axis (in the
        # margins).
        if rect.get_x() > mdates.date2num(xaxis_lowlimit):
            # First group of bars (qpf)
            if len(rects)/2 > rect_cnt:
                ax2.text(rect.get_x() + rect.get_width() / 2., label_position,
                        f'{str(round(label,2))}',
                        ha='center', va='bottom', color='darkgreen')

            # Second group of bars (frozen qpf)
            else:
                label_position = height - (y_height * 0.07)
                if height > 0.3:
                    ax2.text(rect.get_x() + rect.get_width() / 2., label_position,
                             f'{str(round(label, 2))}',
                             ha='center', va='bottom', color='blue')


    ax1.set_zorder(ax2.get_zorder() + 1)  # put ax1 in front of ax2
    #ax1.xaxis.set_major_locator(ticker.MultipleLocator(xaxis_tick_spacing))
    ax1.patch.set_visible(False)  # hide the 'canvas'
    fig.autofmt_xdate()
    ax1.tick_params(axis='x', rotation=45)

    #plt.savefig(os.path.join(imgdir, 'qpf_graph.png'))
    plt.show()
    return

def sql_create_table(df):
    db_path = os.path.join(os.path.dirname(__file__), 'db.sqlite3')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    createtable = False
    if createtable:
        c.execute("CREATE TABLE IF NOT EXISTS SNODAS("
                  "id INTEGER,"
                  "date DATETIME,"
                  "TotalAF_HellHole FLOAT,"
                  "TotalAF_FrenchMeadows FLOAT,"
                  "TotalAF_SMUD FLOAT,"
                  "AveSWE_SMUD FLOAT,"
                  "AveSWE_HellHole FLOAT,"
                  "AveSWE_FrenchMeadows FLOAT,"
                  "Combined_AF FLOAT,"
                  "obs_date DATETIME,"
                  "UNIQUE(date) ON CONFLICT REPLACE)")
    df = df[~df.index.duplicated()]
    idx = pd.date_range(start='01-01-2004',end=datetime.now())
    df = df.reindex(idx, fill_value = np.NaN)
    df = df[['Date_x', 'TotalAF', 'AveSWE', 'Snowpack', 'AveSWE_HellHole',
             'AveSWE_SMUD','TotalAF_SMUD','TotalAF_HellHole']]
    df.rename(columns={'Date_x': 'obs_date',
                       'TotalAF': 'TotalAF_FrenchMeadows',
                       'AveSWE': 'AveSWE_FrenchMeadows',
                       'Snowpack': 'Combined_AF'}, inplace=True)

    df.to_sql('SNODAS', conn, index=True, if_exists='append', index_label='date')



    return
if __name__ == "__main__":
    #sql_create_table()
    main()
