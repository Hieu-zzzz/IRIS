/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Stepper Motor A4988 x4 - STM32F103C8T6
  *                   Motor1: PA0 = EN, PA1 = STEP, PA2 = DIR
  *                   Motor2: PA3 = EN, PA4 = STEP, PA5 = DIR
  *                   Motor3: PA12 = EN, PA11 = STEP, PA8 = DIR
  *                   Motor4: PB0 = EN, PB1 = STEP, PB8 = DIR
  *                   TIM1: PWM dong co encoder 1 (PB13) va encoder 2 (PB14)
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
volatile int32_t g_encoder_count = 0;
volatile uint8_t g_encoder_last_state = 0;
uint8_t g_encoder_motor1_dir_cmd = 0;
uint8_t g_encoder_motor2_dir_cmd = 0;
uint32_t g_encoder_dir_toggle_tick = 0;

volatile uint8_t g_uart_rx_byte = 0;
volatile uint8_t g_uart_cmd_ready = 0;
char g_uart_rx_line[96] = {0};
char g_uart_cmd_line[96] = {0};
uint8_t g_uart_rx_len = 0;

uint8_t g_stepper_enabled[4] = {0, 0, 0, 0};
uint8_t g_stepper_dir[4] = {0, 0, 0, 0};
uint8_t g_dc_enabled[2] = {0, 0};
uint8_t g_dc_dir[2] = {0, 0};
uint16_t g_dc_pwm_permille[2] = {0, 0};

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
void StepMotor_Enable(uint8_t motor_id, uint8_t enable);
void StepMotor_Step(uint8_t motor_id, uint32_t steps, uint8_t direction, uint32_t delay_us);
void StepMotor_SetDirection(uint8_t motor_id, uint8_t direction);
void StepMotor_StepSync3(uint32_t steps, uint8_t dir_m1, uint8_t dir_m2, uint8_t dir_m3, uint32_t delay_us);
void StepMotor_StepSync4(uint32_t steps, uint8_t dir_m1, uint8_t dir_m2, uint8_t dir_m3, uint8_t dir_m4, uint32_t delay_us);
void EncoderMotor_Init(void);
void EncoderMotor_SetEnable(uint8_t enable);
void EncoderMotor_SetDirection(uint8_t direction);
void EncoderMotor_SetDirectionSmooth(uint8_t direction, uint16_t step_permille, uint32_t step_delay_ms);
void EncoderMotor_SetPwmPermille(uint16_t pwm_permille);
void EncoderMotor_SetPwmPermilleSmooth(uint16_t target_pwm_permille, uint16_t step_permille, uint32_t step_delay_ms);
void EncoderMotor1_Init(void);
void EncoderMotor1_SetEnable(uint8_t enable);
void EncoderMotor1_SetDirection(uint8_t direction);
void EncoderMotor1_SetDirectionSmooth(uint8_t direction, uint16_t step_permille, uint32_t step_delay_ms);
void EncoderMotor1_SetPwmPermille(uint16_t pwm_permille);
void EncoderMotor1_SetPwmPermilleSmooth(uint16_t target_pwm_permille, uint16_t step_permille, uint32_t step_delay_ms);
void Encoder_Init(void);
int32_t Encoder_GetCount(void);
void Encoder_Reset(void);
void Delay_us_Init(void);
void Delay_us(uint32_t us);
void UartProtocol_Init(void);
void UartProtocol_Poll(void);
void UartProtocol_SendStatus(void);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_TIM1_Init();
  MX_USART1_UART_Init();
  /* USER CODE BEGIN 2 */

  // Khoi tao Delay_us bang bo dem chu ky CPU (DWT)
  Delay_us_Init();

  // Khoi tao trang thai ban dau cua encoder
  Encoder_Init();

  // Khoi tao dong co encoder 1 (PWM + DIR)
  EncoderMotor1_Init();
  EncoderMotor1_SetEnable(0);
  EncoderMotor1_SetDirection(0);
  EncoderMotor1_SetPwmPermille(0);

  // Khoi tao dong co encoder 2 (PWM + DIR)
  EncoderMotor_Init();
  EncoderMotor_SetEnable(0);
  EncoderMotor_SetDirection(0);
  EncoderMotor_SetPwmPermille(0);

  // Enable 4 dong co
  StepMotor_Enable(1, 1);
  StepMotor_Enable(2, 1);
  StepMotor_Enable(3, 1);
  StepMotor_Enable(4, 1);
  g_stepper_enabled[0] = 1U;
  g_stepper_enabled[1] = 1U;
  g_stepper_enabled[2] = 1U;
  g_stepper_enabled[3] = 1U;
  g_stepper_dir[0] = 0U;
  g_stepper_dir[1] = 0U;
  g_stepper_dir[2] = 0U;
  g_stepper_dir[3] = 0U;

  g_dc_enabled[0] = 0U;
  g_dc_enabled[1] = 0U;
  g_dc_dir[0] = 0U;
  g_dc_dir[1] = 0U;
  g_dc_pwm_permille[0] = 0U;
  g_dc_pwm_permille[1] = 0U;

  UartProtocol_Init();
  HAL_Delay(100);

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    UartProtocol_Poll();
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

static uint16_t g_encoder_motor1_pwm_permille = 0;
static uint16_t g_encoder_motor2_pwm_permille = 0;
static uint8_t g_encoder_motor1_direction = 0;
static uint8_t g_encoder_motor2_direction = 0;

static void EncoderMotor_InternalSetPwm(TIM_HandleTypeDef *htim, uint32_t channel, uint16_t pwm_permille)
{
  uint32_t period = __HAL_TIM_GET_AUTORELOAD(htim);
  uint32_t pulse;

  if(pwm_permille > 1000U)
  {
    pwm_permille = 1000U;
  }

  pulse = (period * pwm_permille) / 1000U;
  __HAL_TIM_SET_COMPARE(htim, channel, pulse);
}

void EncoderMotor1_Init(void)
{
  HAL_TIMEx_PWMN_Start(&htim1, TIM_CHANNEL_1);
  EncoderMotor1_SetPwmPermille(0);
}

void EncoderMotor1_SetEnable(uint8_t enable)
{
  HAL_GPIO_WritePin(ENC_MOTOR1_PORT, ENC_MOTOR1_EN_PIN, enable ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

void EncoderMotor1_SetDirection(uint8_t direction)
{
  HAL_GPIO_WritePin(ENC_MOTOR1_PORT, ENC_MOTOR1_DIR_PIN, direction ? GPIO_PIN_SET : GPIO_PIN_RESET);
  g_encoder_motor1_direction = direction ? 1U : 0U;
}

void EncoderMotor1_SetDirectionSmooth(uint8_t direction, uint16_t step_permille, uint32_t step_delay_ms)
{
  uint16_t keep_pwm;
  uint8_t target_direction = direction ? 1U : 0U;

  if(target_direction == g_encoder_motor1_direction)
  {
    return;
  }

  keep_pwm = g_encoder_motor1_pwm_permille;
  EncoderMotor1_SetPwmPermilleSmooth(0U, step_permille, step_delay_ms);
  EncoderMotor1_SetDirection(target_direction);
  EncoderMotor1_SetPwmPermilleSmooth(keep_pwm, step_permille, step_delay_ms);
}

void EncoderMotor1_SetPwmPermille(uint16_t pwm_permille)
{
  if(pwm_permille > 1000U)
  {
    pwm_permille = 1000U;
  }
  EncoderMotor_InternalSetPwm(&htim1, TIM_CHANNEL_1, pwm_permille);
  g_encoder_motor1_pwm_permille = pwm_permille;
}

void EncoderMotor1_SetPwmPermilleSmooth(uint16_t target_pwm_permille, uint16_t step_permille, uint32_t step_delay_ms)
{
  uint16_t current_pwm;

  if(target_pwm_permille > 1000U)
  {
    target_pwm_permille = 1000U;
  }

  if(step_permille == 0U)
  {
    step_permille = 1U;
  }

  current_pwm = g_encoder_motor1_pwm_permille;

  if(current_pwm < target_pwm_permille)
  {
    while(current_pwm < target_pwm_permille)
    {
      uint32_t next_pwm = current_pwm + step_permille;
      if(next_pwm > target_pwm_permille)
      {
        next_pwm = target_pwm_permille;
      }
      EncoderMotor1_SetPwmPermille((uint16_t)next_pwm);
      current_pwm = (uint16_t)next_pwm;
      HAL_Delay(step_delay_ms);
    }
  }
  else
  {
    while(current_pwm > target_pwm_permille)
    {
      uint32_t next_pwm = (current_pwm > step_permille) ? (current_pwm - step_permille) : 0U;
      if(next_pwm < target_pwm_permille)
      {
        next_pwm = target_pwm_permille;
      }
      EncoderMotor1_SetPwmPermille((uint16_t)next_pwm);
      current_pwm = (uint16_t)next_pwm;
      HAL_Delay(step_delay_ms);
    }
  }
}

void EncoderMotor_Init(void)
{
  HAL_TIMEx_PWMN_Start(&htim1, TIM_CHANNEL_2);
  EncoderMotor_SetPwmPermille(0);
}

void EncoderMotor_SetEnable(uint8_t enable)
{
  HAL_GPIO_WritePin(ENC_MOTOR_PORT, ENC_MOTOR_EN_PIN, enable ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

void EncoderMotor_SetDirection(uint8_t direction)
{
  HAL_GPIO_WritePin(ENC_MOTOR_PORT, ENC_MOTOR_DIR_PIN, direction ? GPIO_PIN_SET : GPIO_PIN_RESET);
  g_encoder_motor2_direction = direction ? 1U : 0U;
}

void EncoderMotor_SetDirectionSmooth(uint8_t direction, uint16_t step_permille, uint32_t step_delay_ms)
{
  uint16_t keep_pwm;
  uint8_t target_direction = direction ? 1U : 0U;

  if(target_direction == g_encoder_motor2_direction)
  {
    return;
  }

  keep_pwm = g_encoder_motor2_pwm_permille;
  EncoderMotor_SetPwmPermilleSmooth(0U, step_permille, step_delay_ms);
  EncoderMotor_SetDirection(target_direction);
  EncoderMotor_SetPwmPermilleSmooth(keep_pwm, step_permille, step_delay_ms);
}

void EncoderMotor_SetPwmPermille(uint16_t pwm_permille)
{
  if(pwm_permille > 1000U)
  {
    pwm_permille = 1000U;
  }
  EncoderMotor_InternalSetPwm(&htim1, TIM_CHANNEL_2, pwm_permille);
  g_encoder_motor2_pwm_permille = pwm_permille;
}

void EncoderMotor_SetPwmPermilleSmooth(uint16_t target_pwm_permille, uint16_t step_permille, uint32_t step_delay_ms)
{
  uint16_t current_pwm;

  if(target_pwm_permille > 1000U)
  {
    target_pwm_permille = 1000U;
  }

  if(step_permille == 0U)
  {
    step_permille = 1U;
  }

  current_pwm = g_encoder_motor2_pwm_permille;

  if(current_pwm < target_pwm_permille)
  {
    while(current_pwm < target_pwm_permille)
    {
      uint32_t next_pwm = current_pwm + step_permille;
      if(next_pwm > target_pwm_permille)
      {
        next_pwm = target_pwm_permille;
      }
      EncoderMotor_SetPwmPermille((uint16_t)next_pwm);
      current_pwm = (uint16_t)next_pwm;
      HAL_Delay(step_delay_ms);
    }
  }
  else
  {
    while(current_pwm > target_pwm_permille)
    {
      uint32_t next_pwm = (current_pwm > step_permille) ? (current_pwm - step_permille) : 0U;
      if(next_pwm < target_pwm_permille)
      {
        next_pwm = target_pwm_permille;
      }
      EncoderMotor_SetPwmPermille((uint16_t)next_pwm);
      current_pwm = (uint16_t)next_pwm;
      HAL_Delay(step_delay_ms);
    }
  }
}

void Delay_us_Init(void)
{
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

void Encoder_Init(void)
{
  uint8_t a = (HAL_GPIO_ReadPin(ENC_PORT, ENC_A_PIN) == GPIO_PIN_SET) ? 1U : 0U;
  uint8_t b = (HAL_GPIO_ReadPin(ENC_PORT, ENC_B_PIN) == GPIO_PIN_SET) ? 1U : 0U;
  g_encoder_last_state = (uint8_t)((a << 1) | b);
}

int32_t Encoder_GetCount(void)
{
  return g_encoder_count;
}

void Encoder_Reset(void)
{
  g_encoder_count = 0;
}

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
  if((GPIO_Pin == ENC_A_PIN) || (GPIO_Pin == ENC_B_PIN))
  {
    uint8_t a = (HAL_GPIO_ReadPin(ENC_PORT, ENC_A_PIN) == GPIO_PIN_SET) ? 1U : 0U;
    uint8_t b = (HAL_GPIO_ReadPin(ENC_PORT, ENC_B_PIN) == GPIO_PIN_SET) ? 1U : 0U;
    uint8_t current_state = (uint8_t)((a << 1) | b);
    uint8_t transition = (uint8_t)((g_encoder_last_state << 2) | current_state);

    // Bang chuyen trang thai encoder quadrature
    switch(transition)
    {
      case 0x1:
      case 0x7:
      case 0xE:
      case 0x8:
        g_encoder_count++;
        break;

      case 0x2:
      case 0x4:
      case 0xD:
      case 0xB:
        g_encoder_count--;
        break;

      default:
        // Bo qua transition khong hop le (nhieu nhieu hoac noise)
        break;
    }

    g_encoder_last_state = current_state;
  }
}

/**
  * @brief  Bat/tat driver dong co buoc
  * @param  enable: 1 = enable (cho phep dong co hoat dong), 0 = disable (khoa dong co)
  * @retval None
  * @note   Hau het driver: EN = LOW (0) de enable, EN = HIGH (1) de disable
  *         Delay microsecond dung bo dem chu ky CPU (DWT)
  */
void Delay_us(uint32_t us)
{
    uint32_t start = DWT->CYCCNT;
    uint32_t ticks = us * (HAL_RCC_GetHCLKFreq() / 1000000U);
    while((DWT->CYCCNT - start) < ticks);
}

static GPIO_TypeDef* StepMotor_GetPort(uint8_t motor_id)
{
  if(motor_id == 4)
  {
    return M4_PORT;
  }
  if(motor_id == 3)
  {
    return M3_PORT;
  }
  if(motor_id == 2)
  {
    return M2_PORT;
  }
  return M1_PORT;
}

static uint16_t StepMotor_GetEnPin(uint8_t motor_id)
{
  if(motor_id == 4)
  {
    return M4_EN_PIN;
  }
  if(motor_id == 3)
  {
    return M3_EN_PIN;
  }
  if(motor_id == 2)
  {
    return M2_EN_PIN;
  }
  return M1_EN_PIN;
}

static uint16_t StepMotor_GetStepPin(uint8_t motor_id)
{
  if(motor_id == 4)
  {
    return M4_STEP_PIN;
  }
  if(motor_id == 3)
  {
    return M3_STEP_PIN;
  }
  if(motor_id == 2)
  {
    return M2_STEP_PIN;
  }
  return M1_STEP_PIN;
}

static uint16_t StepMotor_GetDirPin(uint8_t motor_id)
{
  if(motor_id == 4)
  {
    return M4_DIR_PIN;
  }
  if(motor_id == 3)
  {
    return M3_DIR_PIN;
  }
  if(motor_id == 2)
  {
    return M2_DIR_PIN;
  }
  return M1_DIR_PIN;
}

/**
  * @brief  Bat/tat driver A4988
  *         EN = LOW -> Enable, EN = HIGH -> Disable
  */
void StepMotor_Enable(uint8_t motor_id, uint8_t enable)
{
  GPIO_TypeDef *motor_port = StepMotor_GetPort(motor_id);
  uint16_t en_pin = StepMotor_GetEnPin(motor_id);

    if(enable)
    HAL_GPIO_WritePin(motor_port, en_pin, GPIO_PIN_RESET); // LOW = Enable
    else
    HAL_GPIO_WritePin(motor_port, en_pin, GPIO_PIN_SET);   // HIGH = Disable
}

/**
  * @brief  Dat huong quay
  *         direction: 0 = thuan, 1 = nguoc
  */
  void StepMotor_SetDirection(uint8_t motor_id, uint8_t direction)
{
    GPIO_TypeDef *motor_port = StepMotor_GetPort(motor_id);
    uint16_t dir_pin = StepMotor_GetDirPin(motor_id);

    if(direction)
      HAL_GPIO_WritePin(motor_port, dir_pin, GPIO_PIN_SET);
    else
      HAL_GPIO_WritePin(motor_port, dir_pin, GPIO_PIN_RESET);
}

/**
  * @brief  Quay dong co
  * @param  steps     : so buoc
  * @param  direction : 0 = thuan, 1 = nguoc
  * @param  delay_us  : delay giua cac buoc (us) - cang nho cang nhanh
  *                     Full step: delay_us >= 1000 de motor khong truot buoc
  */
  void StepMotor_Step(uint8_t motor_id, uint32_t steps, uint8_t direction, uint32_t delay_us)
{
    GPIO_TypeDef *motor_port = StepMotor_GetPort(motor_id);
    uint16_t step_pin = StepMotor_GetStepPin(motor_id);
    uint32_t irq_state;

    if(delay_us < 300U)
    {
      delay_us = 300U;
    }

    StepMotor_SetDirection(motor_id, direction);
    Delay_us(100); // Cho DIR on dinh truoc khi step

    for(uint32_t i = 0; i < steps; i++)
    {
      irq_state = __get_PRIMASK();
      __disable_irq();
      HAL_GPIO_WritePin(motor_port, step_pin, GPIO_PIN_SET);
      Delay_us(8);               // Xung STEP HIGH on dinh hon de giam giat
      HAL_GPIO_WritePin(motor_port, step_pin, GPIO_PIN_RESET);
      if(irq_state == 0U)
      {
        __enable_irq();
      }
      Delay_us(delay_us);        // Delay quyet dinh toc do
    }
}

/**
  * @brief  Quay dong thoi 3 dong co
  * @param  steps   : so buoc moi dong co
  * @param  dir_m1  : huong motor 1 (0/1)
  * @param  dir_m2  : huong motor 2 (0/1)
  * @param  dir_m3  : huong motor 3 (0/1)
  * @param  delay_us: delay giua cac buoc (us)
  */
void StepMotor_StepSync3(uint32_t steps, uint8_t dir_m1, uint8_t dir_m2, uint8_t dir_m3, uint32_t delay_us)
{
    GPIO_TypeDef *m1_port = StepMotor_GetPort(1);
    GPIO_TypeDef *m2_port = StepMotor_GetPort(2);
    GPIO_TypeDef *m3_port = StepMotor_GetPort(3);
    uint16_t m1_step_pin = StepMotor_GetStepPin(1);
    uint16_t m2_step_pin = StepMotor_GetStepPin(2);
    uint16_t m3_step_pin = StepMotor_GetStepPin(3);
    uint32_t irq_state;

    if(delay_us < 300U)
    {
      delay_us = 300U;
    }

    StepMotor_SetDirection(1, dir_m1);
    StepMotor_SetDirection(2, dir_m2);
    StepMotor_SetDirection(3, dir_m3);
    Delay_us(100);

    for(uint32_t i = 0; i < steps; i++)
    {
      irq_state = __get_PRIMASK();
      __disable_irq();
      HAL_GPIO_WritePin(m1_port, m1_step_pin, GPIO_PIN_SET);
      HAL_GPIO_WritePin(m2_port, m2_step_pin, GPIO_PIN_SET);
      HAL_GPIO_WritePin(m3_port, m3_step_pin, GPIO_PIN_SET);
      Delay_us(8);

      HAL_GPIO_WritePin(m1_port, m1_step_pin, GPIO_PIN_RESET);
      HAL_GPIO_WritePin(m2_port, m2_step_pin, GPIO_PIN_RESET);
      HAL_GPIO_WritePin(m3_port, m3_step_pin, GPIO_PIN_RESET);
      if(irq_state == 0U)
      {
        __enable_irq();
      }
      Delay_us(delay_us);
    }
}

/**
  * @brief  Quay dong thoi 4 dong co
  * @param  steps   : so buoc moi dong co
  * @param  dir_m1  : huong motor 1 (0/1)
  * @param  dir_m2  : huong motor 2 (0/1)
  * @param  dir_m3  : huong motor 3 (0/1)
  * @param  dir_m4  : huong motor 4 (0/1)
  * @param  delay_us: delay giua cac buoc (us)
  */
void StepMotor_StepSync4(uint32_t steps, uint8_t dir_m1, uint8_t dir_m2, uint8_t dir_m3, uint8_t dir_m4, uint32_t delay_us)
{
    GPIO_TypeDef *m1_port = StepMotor_GetPort(1);
    GPIO_TypeDef *m2_port = StepMotor_GetPort(2);
    GPIO_TypeDef *m3_port = StepMotor_GetPort(3);
    GPIO_TypeDef *m4_port = StepMotor_GetPort(4);
    uint16_t m1_step_pin = StepMotor_GetStepPin(1);
    uint16_t m2_step_pin = StepMotor_GetStepPin(2);
    uint16_t m3_step_pin = StepMotor_GetStepPin(3);
    uint16_t m4_step_pin = StepMotor_GetStepPin(4);
    uint32_t irq_state;

    if(delay_us < 300U)
    {
      delay_us = 300U;
    }

    StepMotor_SetDirection(1, dir_m1);
    StepMotor_SetDirection(2, dir_m2);
    StepMotor_SetDirection(3, dir_m3);
    StepMotor_SetDirection(4, dir_m4);
    Delay_us(100);

    for(uint32_t i = 0; i < steps; i++)
    {
      irq_state = __get_PRIMASK();
      __disable_irq();
      HAL_GPIO_WritePin(m1_port, m1_step_pin, GPIO_PIN_SET);
      HAL_GPIO_WritePin(m2_port, m2_step_pin, GPIO_PIN_SET);
      HAL_GPIO_WritePin(m3_port, m3_step_pin, GPIO_PIN_SET);
      HAL_GPIO_WritePin(m4_port, m4_step_pin, GPIO_PIN_SET);
      Delay_us(8);

      HAL_GPIO_WritePin(m1_port, m1_step_pin, GPIO_PIN_RESET);
      HAL_GPIO_WritePin(m2_port, m2_step_pin, GPIO_PIN_RESET);
      HAL_GPIO_WritePin(m3_port, m3_step_pin, GPIO_PIN_RESET);
      HAL_GPIO_WritePin(m4_port, m4_step_pin, GPIO_PIN_RESET);
      if(irq_state == 0U)
      {
        __enable_irq();
      }
      Delay_us(delay_us);
    }
}

static void UartProtocol_SendLine(const char *line)
{
  if(line == NULL)
  {
    return;
  }
  HAL_UART_Transmit(&huart1, (uint8_t *)line, (uint16_t)strlen(line), 100);
}

static void UartProtocol_SendOk(const char *msg)
{
  char tx[96];
  if(msg == NULL)
  {
    msg = "OK";
  }
  snprintf(tx, sizeof(tx), "OK %s\r\n", msg);
  UartProtocol_SendLine(tx);
}

static void UartProtocol_SendErr(const char *msg)
{
  char tx[96];
  if(msg == NULL)
  {
    msg = "ERR";
  }
  snprintf(tx, sizeof(tx), "ERR %s\r\n", msg);
  UartProtocol_SendLine(tx);
}

static void UartProtocol_SendHelp(void)
{
  UartProtocol_SendLine("OK CMDS PING|STATUS|ENC_GET|ENC_RST|STEP id steps dir delay_us|STEP_EN id en|STEP_DIR id dir|STEPSYNC4 steps d1 d2 d3 d4 delay_us|DC_EN id(1..2) en|DC_DIR id(1..2) dir|DC_PWM id(1..2) 0..1000\r\n");
}

void UartProtocol_SendStatus(void)
{
  char tx[220];
  long enc = (long)Encoder_GetCount();

  snprintf(tx, sizeof(tx),
           "STATUS ENC=%ld STEP_EN=%u,%u,%u,%u STEP_DIR=%u,%u,%u,%u DC_EN=%u,%u DC_DIR=%u,%u DC_PWM=%u,%u\r\n",
           enc,
           g_stepper_enabled[0], g_stepper_enabled[1], g_stepper_enabled[2], g_stepper_enabled[3],
           g_stepper_dir[0], g_stepper_dir[1], g_stepper_dir[2], g_stepper_dir[3],
           g_dc_enabled[0], g_dc_enabled[1],
           g_dc_dir[0], g_dc_dir[1],
           g_dc_pwm_permille[0], g_dc_pwm_permille[1]);
  UartProtocol_SendLine(tx);
}

static uint8_t UartProtocol_ParseMotorId(uint8_t id, uint8_t max_id)
{
  if((id == 0U) || (id > max_id))
  {
    return 0U;
  }
  return 1U;
}

static void UartProtocol_ApplyDcEnable(uint8_t id, uint8_t enable)
{
  if(id == 1U)
  {
    EncoderMotor1_SetEnable(enable);
  }
  else if(id == 2U)
  {
    EncoderMotor_SetEnable(enable);
  }
  g_dc_enabled[id - 1U] = enable ? 1U : 0U;
}

static void UartProtocol_ApplyDcDirection(uint8_t id, uint8_t direction)
{
  if(id == 1U)
  {
    EncoderMotor1_SetDirectionSmooth(direction, 10U, 2U);
  }
  else if(id == 2U)
  {
    EncoderMotor_SetDirectionSmooth(direction, 10U, 2U);
  }
  g_dc_dir[id - 1U] = direction ? 1U : 0U;
}

static void UartProtocol_ApplyDcPwm(uint8_t id, uint16_t pwm)
{
  if(id == 1U)
  {
    EncoderMotor1_SetPwmPermilleSmooth(pwm, 10U, 2U);
  }
  else if(id == 2U)
  {
    EncoderMotor_SetPwmPermilleSmooth(pwm, 10U, 2U);
  }
  g_dc_pwm_permille[id - 1U] = pwm;
}

static void UartProtocol_ProcessCommand(const char *line)
{
  uint8_t id;
  uint8_t v1;
  uint8_t v2;
  uint8_t v3;
  uint8_t v4;
  unsigned long steps;
  unsigned long delay_us;
  uint16_t pwm;

  if((line == NULL) || (line[0] == '\0'))
  {
    return;
  }

  if(strcmp(line, "PING") == 0)
  {
    UartProtocol_SendOk("PONG");
    return;
  }

  if(strcmp(line, "HELP") == 0)
  {
    UartProtocol_SendHelp();
    return;
  }

  if(strcmp(line, "STATUS") == 0)
  {
    UartProtocol_SendStatus();
    return;
  }

  if(strcmp(line, "ENC_GET") == 0)
  {
    UartProtocol_SendStatus();
    return;
  }

  if(strcmp(line, "ENC_RST") == 0)
  {
    Encoder_Reset();
    UartProtocol_SendOk("ENC_RST");
    return;
  }

  if(sscanf(line, "STEP_EN %hhu %hhu", &id, &v1) == 2)
  {
    if(!UartProtocol_ParseMotorId(id, 4U))
    {
      UartProtocol_SendErr("STEP_ID");
      return;
    }
    StepMotor_Enable(id, v1 ? 1U : 0U);
    g_stepper_enabled[id - 1U] = v1 ? 1U : 0U;
    UartProtocol_SendOk("STEP_EN");
    return;
  }

  if(sscanf(line, "STEP_DIR %hhu %hhu", &id, &v1) == 2)
  {
    if(!UartProtocol_ParseMotorId(id, 4U))
    {
      UartProtocol_SendErr("STEP_ID");
      return;
    }
    StepMotor_SetDirection(id, v1 ? 1U : 0U);
    g_stepper_dir[id - 1U] = v1 ? 1U : 0U;
    UartProtocol_SendOk("STEP_DIR");
    return;
  }

  if(sscanf(line, "STEP %hhu %lu %hhu %lu", &id, &steps, &v1, &delay_us) == 4)
  {
    if(!UartProtocol_ParseMotorId(id, 4U))
    {
      UartProtocol_SendErr("STEP_ID");
      return;
    }
    if(delay_us < 100UL)
    {
      delay_us = 100UL;
    }
    StepMotor_Step(id, (uint32_t)steps, v1 ? 1U : 0U, (uint32_t)delay_us);
    g_stepper_dir[id - 1U] = v1 ? 1U : 0U;
    UartProtocol_SendOk("STEP");
    return;
  }

  if(sscanf(line, "STEPSYNC4 %lu %hhu %hhu %hhu %hhu %lu", &steps, &v1, &v2, &v3, &v4, &delay_us) == 6)
  {
    if(delay_us < 100UL)
    {
      delay_us = 100UL;
    }
    StepMotor_StepSync4((uint32_t)steps, v1 ? 1U : 0U, v2 ? 1U : 0U, v3 ? 1U : 0U, v4 ? 1U : 0U, (uint32_t)delay_us);
    g_stepper_dir[0] = v1 ? 1U : 0U;
    g_stepper_dir[1] = v2 ? 1U : 0U;
    g_stepper_dir[2] = v3 ? 1U : 0U;
    g_stepper_dir[3] = v4 ? 1U : 0U;
    UartProtocol_SendOk("STEPSYNC4");
    return;
  }

  if(sscanf(line, "DC_EN %hhu %hhu", &id, &v1) == 2)
  {
    if(!UartProtocol_ParseMotorId(id, 2U))
    {
      UartProtocol_SendErr("DC_ID");
      return;
    }
    UartProtocol_ApplyDcEnable(id, v1 ? 1U : 0U);
    UartProtocol_SendOk("DC_EN");
    return;
  }

  if(sscanf(line, "DC_DIR %hhu %hhu", &id, &v1) == 2)
  {
    if(!UartProtocol_ParseMotorId(id, 2U))
    {
      UartProtocol_SendErr("DC_ID");
      return;
    }
    UartProtocol_ApplyDcDirection(id, v1 ? 1U : 0U);
    UartProtocol_SendOk("DC_DIR");
    return;
  }

  if(sscanf(line, "DC_PWM %hhu %hu", &id, &pwm) == 2)
  {
    if(!UartProtocol_ParseMotorId(id, 2U))
    {
      UartProtocol_SendErr("DC_ID");
      return;
    }
    if(pwm > 1000U)
    {
      pwm = 1000U;
    }
    UartProtocol_ApplyDcPwm(id, pwm);
    UartProtocol_SendOk("DC_PWM");
    return;
  }

  if(sscanf(line, "STEPDIR4 %hhu %hhu %hhu %hhu", &v1, &v2, &v3, &v4) == 4)
  {
    StepMotor_SetDirection(1U, v1 ? 1U : 0U);
    StepMotor_SetDirection(2U, v2 ? 1U : 0U);
    StepMotor_SetDirection(3U, v3 ? 1U : 0U);
    StepMotor_SetDirection(4U, v4 ? 1U : 0U);
    g_stepper_dir[0] = v1 ? 1U : 0U;
    g_stepper_dir[1] = v2 ? 1U : 0U;
    g_stepper_dir[2] = v3 ? 1U : 0U;
    g_stepper_dir[3] = v4 ? 1U : 0U;
    UartProtocol_SendOk("STEPDIR4");
    return;
  }

  UartProtocol_SendErr("UNKNOWN");
}

void UartProtocol_Init(void)
{
  g_uart_rx_len = 0U;
  g_uart_cmd_ready = 0U;
  HAL_UART_Receive_IT(&huart1, (uint8_t *)&g_uart_rx_byte, 1U);
  UartProtocol_SendLine("OK READY UART1 115200\r\n");
  UartProtocol_SendHelp();
  UartProtocol_SendStatus();
}

void UartProtocol_Poll(void)
{
  static uint32_t last_status_tick = 0U;
  static int32_t last_encoder = 0;
  static uint32_t last_encoder_push_tick = 0U;
  int32_t encoder_now;

  if(g_uart_cmd_ready)
  {
    __disable_irq();
    g_uart_cmd_ready = 0U;
    __enable_irq();
    UartProtocol_ProcessCommand(g_uart_cmd_line);
  }

  if((HAL_GetTick() - last_status_tick) >= 200U)
  {
    last_status_tick = HAL_GetTick();
    UartProtocol_SendStatus();
  }

  encoder_now = Encoder_GetCount();
  if((encoder_now != last_encoder) && ((HAL_GetTick() - last_encoder_push_tick) >= 40U))
  {
    last_encoder = encoder_now;
    last_encoder_push_tick = HAL_GetTick();
    UartProtocol_SendStatus();
  }
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  if(huart->Instance == USART1)
  {
    uint8_t c = g_uart_rx_byte;

    if((c == '\n') || (c == '\r'))
    {
      if((g_uart_rx_len > 0U) && (g_uart_cmd_ready == 0U))
      {
        g_uart_rx_line[g_uart_rx_len] = '\0';
        strncpy(g_uart_cmd_line, g_uart_rx_line, sizeof(g_uart_cmd_line) - 1U);
        g_uart_cmd_line[sizeof(g_uart_cmd_line) - 1U] = '\0';
        g_uart_cmd_ready = 1U;
      }
      g_uart_rx_len = 0U;
    }
    else if(g_uart_rx_len < (sizeof(g_uart_rx_line) - 1U))
    {
      g_uart_rx_line[g_uart_rx_len++] = (char)c;
    }
    else
    {
      g_uart_rx_len = 0U;
      UartProtocol_SendErr("LINE_TOO_LONG");
    }

    HAL_UART_Receive_IT(&huart1, (uint8_t *)&g_uart_rx_byte, 1U);
  }
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
  if(huart->Instance == USART1)
  {
    __HAL_UART_CLEAR_OREFLAG(huart);
    HAL_UART_Receive_IT(&huart1, (uint8_t *)&g_uart_rx_byte, 1U);
  }
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
