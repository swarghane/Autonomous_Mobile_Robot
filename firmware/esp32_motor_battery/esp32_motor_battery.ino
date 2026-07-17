#define L_RPWM 2
#define L_LPWM 3
#define R_RPWM 4
#define R_LPWM 5
#define L_REN 6
#define L_LEN 7
#define R_REN 8
#define R_LEN 9

// ── Battery monitor ────────────────────────────────────
// Divider: BATT+ -- R1(100k) -- A1 -- R2(33k) -- GND
const int BATT_PIN = A1;
const unsigned long BATT_SEND_INTERVAL_MS = 2000;
unsigned long lastBattSend = 0;

void setup(){
  Serial.begin(115200);
  pinMode(L_RPWM,OUTPUT);
  pinMode(L_LPWM,OUTPUT);
  pinMode(R_RPWM,OUTPUT);
  pinMode(R_LPWM,OUTPUT);
  pinMode(L_REN,OUTPUT);
  pinMode(L_LEN,OUTPUT);
  pinMode(R_REN,OUTPUT);
  pinMode(R_LEN,OUTPUT);
  digitalWrite(L_REN,HIGH);
  digitalWrite(L_LEN,HIGH);
  digitalWrite(R_REN,HIGH);
  digitalWrite(R_LEN,HIGH);
  stopMotors();

  analogReadResolution(12);   // 0-4095, matches motor_control_node.py's assumption
}

void loop(){
  // ── Motor commands (unchanged) ───────────────────────
  if(Serial.available()){
    String data=Serial.readStringUntil('\n');
    int commaIndex=data.indexOf(',');
    if(commaIndex>0){
      int leftSpeed=data.substring(0,commaIndex).toInt();
      int rightSpeed=data.substring(commaIndex+1).toInt();
      leftSpeed=constrain(leftSpeed,-255,255);
      rightSpeed=constrain(rightSpeed,-255,255);
      controlMotor(leftSpeed,rightSpeed);
    }
  }

  // ── Battery report (non-blocking, every 2s) ──────────
  unsigned long now = millis();
  if (now - lastBattSend >= BATT_SEND_INTERVAL_MS) {
    lastBattSend = now;
    int raw = analogRead(BATT_PIN);
    Serial.print("BATT:");
    Serial.println(raw);
  }
}

void controlMotor(int leftSpeed,int rightSpeed){
  // LEFT MOTOR
  if(leftSpeed>0){
    analogWrite(L_RPWM,leftSpeed);
    analogWrite(L_LPWM,0);
  }
  else if(leftSpeed<0){
    analogWrite(L_RPWM,0);
    analogWrite(L_LPWM,abs(leftSpeed));
  }
  else{
    analogWrite(L_RPWM,0);
    analogWrite(L_LPWM,0);
  }
  // RIGHT MOTOR
  if(rightSpeed>0){
    analogWrite(R_RPWM,rightSpeed);
    analogWrite(R_LPWM,0);
  }
  else if(rightSpeed<0){
    analogWrite(R_RPWM,0);
    analogWrite(R_LPWM,abs(rightSpeed));
  }
  else{
    analogWrite(R_RPWM,0);
    analogWrite(R_LPWM,0);
  }
}

void stopMotors(){
  analogWrite(L_RPWM,0);
  analogWrite(L_LPWM,0);
  analogWrite(R_RPWM,0);
  analogWrite(R_LPWM,0);
}
