idf_component_register(SRCS "hello_world_main.c"
                       PRIV_REQUIRES esp_driver_gpio esp_driver_ledc esp_driver_uart
                       REQUIRES driver
                       INCLUDE_DIRS ".")
