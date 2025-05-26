## 삼변측량 API 설명
### request
GET


### path
/get_position

### query parameter
up_distance: float, <br>
down_distance: float, <br>
front_distance: float, <br>
left_distance: float, <br>
arm_length: float<br>


### response
{
  "x": float,
  "y": float,
  "z": float
}
