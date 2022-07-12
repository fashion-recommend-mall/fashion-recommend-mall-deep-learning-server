# F**ashion Recommend Mall Deep Learning Server**

![twitter_header_photo_2](https://user-images.githubusercontent.com/66009926/176111438-92f1352f-c063-4e5e-a715-f8ef3cbe0757.png)

## Project Description

This project categorizes the fashion image using the Pytorch model.

First, If you enter a fashion image like the below request, Calculates the weight through the Pytorch model.
```
// Request example
// img_link should be http not https!
curl -X GET "{Deep Learing Server URL}/upload?img_path={img_link}"

// Weight example
{
    "traditional" : 0.123,
    "manish" : -0.123,
    "feminine" : 1.123,
    "ethnic" : 0,
    "contemporary" : 0,
    "natural" : 0,
    "genderless" : 0,
    "sporty" : 0,
    "subculture" : 0,
    "casual" : 0
}
```

Second, It is sorted in the order of high weights.
```
// Weight example
{
    "feminine" : 1.123,
    "traditional" : 0.123,
    "manish" : -0.123,
    "ethnic" : 0,
    "contemporary" : 0,
    "natural" : 0,
    "genderless" : 0,
    "sporty" : 0,
    "subculture" : 0,
    "casual" : 0
}
```

Finally, It returns the two highest categories in the format below.
```
// Response example
{
    "first style" : "feminine",
    "second style" : "traditional",
}

```

Final Request & Response like below

```
// Request
curl -X GET "{Deep Learing Server URL}/upload?img_path={img_link}"

// Response
{
  "first style" : "",
  "second style" : ""
}
```

Got the Pytorch model from the below site

Link: [AI Hube](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=51)

<img width="1440" alt="image" src="https://user-images.githubusercontent.com/66009926/178394650-24d7057d-a8f1-4f07-9d90-676d2fce192b.png">

---

## System Structure

![Slide2](https://user-images.githubusercontent.com/66009926/178388989-d9c858a3-ac12-41cf-b0b3-32581e3dd9af.jpg)

---

## Folder Structure

![Untitled](https://user-images.githubusercontent.com/66009926/176206782-b9d33093-eff6-4431-b485-89a73e2c3786.png)

---

## How to run

### Requirement

```
// Build Requirement
Flask : 2.1.2
Torch : 1.11.0

// Run Requirement
Docker : 20.10.16
```

### Build & Run

```
// Running Server
$ sudo git clone {this repo}
$ sudo docker build -t deep_learning_server .
$ sudo docker run -d -p 3000:3000 deep_learning_server
```

---

## What I learn
