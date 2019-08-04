import urllib.request
import time
import cv2
import predict_female_vs_male as predict

def main():
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    female = 78
    male = 61

    for index_number in range(1, 101):
        downloaded_file = f'download/original_size/{index_number}.png'
        print(f'Downloaded {downloaded_file}')
        url = f'https://thispersondoesnotexist.com/image'
        urllib.request.urlretrieve(url, downloaded_file)
        img = cv2.imread(downloaded_file, cv2.IMREAD_UNCHANGED)

        resized = cv2.resize(img, (200, 200), interpolation = cv2.INTER_AREA)

        prediction = predict.predict_gender(downloaded_file)
        print(prediction)
        if prediction == 'male':
            male = male + 1
            output_file = f'./download/{prediction}/{male}.png'

        else:
            female = female + 1
            output_file = f'./download/{prediction}/{female}.png'

        cv2.imwrite(output_file, resized)
        print(f'Resized and copied {downloaded_file} to {output_file}')
        for seconds in reversed(range(1,10)):
            print(f'Sleeping for {seconds} seconds ...')
            time.sleep(1)


if __name__ == '__main__':
    main();
