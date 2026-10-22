#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/ledc.h"
#include "driver/uart.h"
#include "esp_err.h"

/* ---- Wiring of the HW-478 RGB LED module ---- */
#define PIN_RED    23
#define PIN_GREEN  22
#define PIN_BLUE   21

/* 0 = common cathode (module's common pin goes to GND). Set to 1 ONLY for a common-anode module. */
#define LED_COMMON_ANODE 0

/* ---- OVERCURRENT GUARDRAILS (the module has NO resistors) ----
 * 1. GPIO drive strength is set to the weakest level (about 5 mA) on all three pins.
 * 2. Every channel is limited to MAX_CHANNEL_DUTY_PCT percent PWM duty.
 * 3. All channels together are limited to MAX_TOTAL_DUTY_PCT percent.
 * 4. The ONLY function that writes PWM duty is led_write_channel(), and it is only called from led_set_rgb_safe().
 * 5. LED is OFF at boot, on invalid input, and after FAILSAFE_TIMEOUT_MS without a valid message.
 * 6. The two static asserts below stop the build if someone raises the caps too high.
 * DO NOT raise these numbers. Add real resistors (100 to 220 ohm per color) before raising them. */
#define MAX_CHANNEL_DUTY_PCT 30
#define MAX_TOTAL_DUTY_PCT   60
#define PWM_FREQ_HZ          5000
#define FAILSAFE_TIMEOUT_MS  5000

_Static_assert(MAX_CHANNEL_DUTY_PCT <= 40, "per-channel duty cap too high for a resistor-less LED");
_Static_assert(MAX_TOTAL_DUTY_PCT <= 80, "total duty cap too high for a resistor-less LED");

#define UART_PORT    UART_NUM_0
#define LINE_MAX_LEN 32

static const int LED_GPIO[3] = { PIN_RED, PIN_GREEN, PIN_BLUE };
static const ledc_channel_t LED_CH[3] = { LEDC_CHANNEL_0, LEDC_CHANNEL_1, LEDC_CHANNEL_2 };

/* The ONLY place that writes PWM duty. duty is 0..255. */
static void led_write_channel(int i, uint32_t duty)
{
    if (duty > 255) {
        duty = 255;
    }
    if (LED_COMMON_ANODE) {
        duty = 255 - duty;
    }
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LED_CH[i], duty);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LED_CH[i]);
}

/* r, g, b are wanted brightness 0..255. The caps are applied here. */
static void led_set_rgb_safe(int r, int g, int b)
{
    int v[3] = { r, g, b };
    int total = 0;
    for (int i = 0; i < 3; i++) {
        if (v[i] < 0) {
            v[i] = 0;
        }
        if (v[i] > 255) {
            v[i] = 255;
        }
        v[i] = v[i] * MAX_CHANNEL_DUTY_PCT / 100;
        total += v[i];
    }
    int total_cap = 255 * MAX_TOTAL_DUTY_PCT / 100;
    if (total > total_cap) {
        for (int i = 0; i < 3; i++) {
            v[i] = v[i] * total_cap / total;
        }
    }
    for (int i = 0; i < 3; i++) {
        led_write_channel(i, (uint32_t)v[i]);
    }
}

static void led_off(void)
{
    led_set_rgb_safe(0, 0, 0);
}

static void led_init(void)
{
    for (int i = 0; i < 3; i++) {
        gpio_reset_pin(LED_GPIO[i]);
        gpio_set_direction(LED_GPIO[i], GPIO_MODE_OUTPUT);
        gpio_set_level(LED_GPIO[i], LED_COMMON_ANODE ? 1 : 0);
        gpio_set_drive_capability(LED_GPIO[i], GPIO_DRIVE_CAP_0);
    }

    ledc_timer_config_t timer = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .duty_resolution = LEDC_TIMER_8_BIT,
        .timer_num = LEDC_TIMER_0,
        .freq_hz = PWM_FREQ_HZ,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&timer));

    for (int i = 0; i < 3; i++) {
        ledc_channel_config_t ch = {
            .gpio_num = LED_GPIO[i],
            .speed_mode = LEDC_LOW_SPEED_MODE,
            .channel = LED_CH[i],
            .intr_type = LEDC_INTR_DISABLE,
            .timer_sel = LEDC_TIMER_0,
            .duty = LED_COMMON_ANODE ? 255 : 0,
            .hpoint = 0,
        };
        ESP_ERROR_CHECK(ledc_channel_config(&ch));
        gpio_set_drive_capability(LED_GPIO[i], GPIO_DRIVE_CAP_0);
    }

    led_off();
}

/* stress 0..100 -> green (0) -> yellow (50) -> red (100). Blue stays 0. */
static void stress_to_rgb(float s, int *r, int *g, int *b)
{
    if (s < 0.0f) {
        s = 0.0f;
    }
    if (s > 100.0f) {
        s = 100.0f;
    }
    if (s <= 50.0f) {
        *r = (int)(s / 50.0f * 255.0f);
        *g = 255;
    } else {
        *r = 255;
        *g = (int)((100.0f - s) / 50.0f * 255.0f);
    }
    *b = 0;
}

static void handle_line(const char *line, TickType_t *last_msg_tick)
{
    if (strcmp(line, "OFF") == 0) {
        led_off();
        *last_msg_tick = xTaskGetTickCount();
        return;
    }
    if (strncmp(line, "S:", 2) == 0) {
        char *end = NULL;
        float s = strtof(line + 2, &end);
        if (end != line + 2 && isfinite(s)) {
            int r, g, b;
            stress_to_rgb(s, &r, &g, &b);
            led_set_rgb_safe(r, g, b);
            *last_msg_tick = xTaskGetTickCount();
            return;
        }
    }
    led_off();   /* anything invalid -> LED off */
}

void app_main(void)
{
    led_init();
    ESP_ERROR_CHECK(uart_driver_install(UART_PORT, 256, 0, 0, NULL, 0));
    printf("LED controller ready. Waiting for lines S:<stress> or OFF.\n");

    char line[LINE_MAX_LEN];
    int len = 0;
    TickType_t last_msg_tick = xTaskGetTickCount();

    while (1) {
        uint8_t buf[64];
        int n = uart_read_bytes(UART_PORT, buf, sizeof(buf), pdMS_TO_TICKS(100));
        for (int i = 0; i < n; i++) {
            char c = (char)buf[i];
            if (c == '\n') {
                line[len] = '\0';
                handle_line(line, &last_msg_tick);
                len = 0;
            } else if (c != '\r') {
                if (len < LINE_MAX_LEN - 1) {
                    line[len++] = c;
                } else {
                    len = 0;   /* overlong line: discard it */
                }
            }
        }
        if ((xTaskGetTickCount() - last_msg_tick) > pdMS_TO_TICKS(FAILSAFE_TIMEOUT_MS)) {
            led_off();
        }
    }
}
