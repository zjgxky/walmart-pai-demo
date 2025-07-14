CREATE TABLE IF NOT EXISTS walmart_sales_raw (
    store BIGINT COMMENT '门店ID',
    date_str STRING COMMENT '日期字符串',
    weekly_sales DOUBLE COMMENT '周销量',
    holiday_flag BIGINT COMMENT '节假日标识 0/1',
    temperature DOUBLE COMMENT '温度',
    fuel_price DOUBLE COMMENT '燃油价格',
    cpi DOUBLE COMMENT '消费者价格指数',
    unemployment DOUBLE COMMENT '失业率'
) 
COMMENT 'Walmart原始销售数据表';