package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"time"
)

func MakeRequest(url string, ch chan<- string) {
	start := time.Now()
	resp, _ := http.Get(url)
	secs := time.Since(start).Seconds()
	body, _ := ioutil.ReadAll(resp.Body)
	ch <- fmt.Sprintf("%.2f elapsed with response length: %s %s", secs, body, url)
}
func main() {
	start := time.Now()
	// ch := make(chan string)
	ch1 := make(chan string)
	for i := 0; i <= 100; i++ {
		// wait for 8 milliseconds to simulate the patient
	        time.Sleep(8 * time.Millisecond)
		go MakeRequest("http://127.0.0.1:8000/RayServeProfile/ECG", ch1)
	}
	for i := 0; i <= 100; i++ {
		fmt.Println(<-ch1)
	}
	ch := make(chan string)
	for i := 0; i <= 3800; i++ {
		// wait for 8 milliseconds to simulate the patient
		// incoming data
		time.Sleep(8 * time.Millisecond)
		go MakeRequest("http://127.0.0.1:8000/RayServeProfile/ECG", ch)
	}
	for i := 0; i <= 3800; i++ {
		fmt.Println(<-ch)
	}
	fmt.Printf("%.2fs elapsed\n", time.Since(start).Seconds())
}
