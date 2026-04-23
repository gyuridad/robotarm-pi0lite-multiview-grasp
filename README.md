## 실행 영상(4배속)
  https://drive.google.com/file/d/1QyYSzDktjU-YoII_cun-H9NawKZrPH7U/view?usp=drive_link 

## 노션 요약
  https://www.notion.so/_1_6-Pi0-VLA-34b5d0bd69638013bba5c1bcd6c57af7?source=copy_link 

# robotarm-pi0lite-multiview-grasp

## 프로젝트 개요

이 프로젝트는 ROS 2, MoveIt2, Isaac Sim, 멀티뷰 카메라 입력, 그리고 `pi0_lite` 기반 정책을 결합하여 로봇팔이 의자를 인식하고 접근한 뒤 grasp와 lift까지 수행하도록 구성한 end-to-end manipulation pipeline입니다.

핵심 실행 로직은 `chair_grasp_moveit_pi0_lite_policy_external_cartesian_standalone.py`에 구현되어 있으며, 다음 과정을 하나의 파이프라인으로 연결합니다.

- 손목 카메라 + 외부 카메라 기반 멀티뷰 입력 구성
- 의자 검출 결과 기반 3D 목표 위치 계산
- MoveIt2 IK/FK 기반 pre-grasp pose 생성
- `pi0_lite_phase_aux_external.py` 기반 cartesian policy 추론
- 정책이 예측한 `cartesian_delta`를 실제 로봇 제어로 변환
- grasp 이후 lift 동작 수행 가능

이 프로젝트에서 가장 중요하게 본 질문은 다음이었습니다.

**손목카메라 단일 이미지로만 학습한 정책은 의자에 충분히 잘 접근하지 못했는데, 외부카메라 이미지를 추가한 멀티뷰 입력이 실제 접근 품질을 개선할 수 있는가?**

초기에는 손목카메라 단일뷰 기반 정책으로 실행했을 때 다음과 같은 한계가 있었습니다.

- 의자까지 충분히 내려오지 못함
- 접근 도중 멈추거나 매우 보수적으로 움직임
- grasp 가능한 높이까지 안정적으로 도달하지 못함
- 손목 시점만으로는 전체 장면과 의자와의 상대 위치를 충분히 안정적으로 파악하지 못함

이를 보완하기 위해 외부카메라 이미지를 추가한 멀티뷰 정책을 구성했고, 실제로 접근 단계의 품질이 이전보다 개선되는지를 중점적으로 확인했습니다.

## 핵심 목표

이 프로젝트의 핵심 목표는 단순히 멀티뷰 입력을 사용하는 것이 아니라,

- 외부카메라 추가가 실제 접근 동작을 더 안정적으로 만드는지
- 의자 가까이까지 내려가는 품질이 개선되는지
- 손목 단일뷰에서 보이던 조기 정지나 접근 부족 문제가 줄어드는지

를 실제 로봇 실행 기준으로 검증하는 것이었습니다.

즉 이 프로젝트는 다음 질문에 대한 실험이라고 볼 수 있습니다.

**멀티뷰 입력이 실제 로봇의 의자 접근 문제를 개선할 수 있는가?**

## 기술 스택

- ROS 2 `rclpy`
- MoveIt2 `GetPositionIK`, `GetPositionFK`
- TF2 `TransformListener`, `Buffer`
- Isaac Sim
- Python
- NumPy
- OpenCV
- PyTorch
- CLIP text encoder
- 멀티뷰 RGB 카메라 입력
- Panda 로봇팔 `JointState` 기반 제어

## 주요 기능

- 손목 카메라 + 외부 카메라 기반 멀티뷰 정책 추론
- 의자 검출 결과 기반 목표 위치 추정
- 카메라 좌표계에서 world 좌표계로의 변환
- MoveIt2 IK/FK를 활용한 pre-grasp / policy target / lift pose 계산
- `pi0_lite_phase_aux_external.py` 학습 가중치를 사용한 cartesian policy inference
- 이미지 + instruction + robot state 기반 policy 추론
- 예측된 `world_vector`, `rotation_delta`를 실제 로봇 motion으로 변환
- phase-aware auxiliary learning 기반 정책 학습
- 데이터 수집, 1차 가공, 2차 가공, 학습, 추론까지 연결된 파이프라인 제공
- 실행 로그 및 policy step 기록 저장

## 시스템 아키텍처

### 1. Perception

손목 카메라와 외부 카메라에서 현재 RGB 이미지를 수신합니다.

- 손목 카메라
  - 로봇 말단 기준의 근접 시점
  - 접촉 직전의 세밀한 정보 제공

- 외부 카메라
  - 장면 전체 구조
  - 로봇팔과 의자의 상대적 위치
  - 접근 방향과 전체 정렬 정보 제공

또한 `/chair_detection_json`을 통해 의자 검출 결과를 받아 목표 물체의 위치를 계산합니다.

### 2. Planning

의자 검출 결과를 바탕으로 다음 위치를 계산합니다.

- goal 위치
- pre-grasp 위치
- lift 위치

이후 MoveIt2 IK/FK 서비스를 이용해

- 현재 EE pose 계산
- 목표 pose에 대한 joint target 계산

을 수행합니다.

즉 정책이 직접 joint 값을 예측하는 것이 아니라, 먼저 손끝 기준 목표 pose를 만들고 이를 IK를 통해 실제 joint motion으로 변환합니다.

### 3. Policy Control

먼저 로봇팔을 pre-grasp pose로 이동시킵니다.  
이후 policy loop를 실행합니다.

policy는 다음 입력을 기반으로 action을 예측합니다.

- 현재 손목 RGB 이미지
- 현재 외부 RGB 이미지
- 자연어 instruction
- 현재 joint / gripper / end-effector state

예측된 action은 cartesian 형식으로 해석됩니다.

- `world_vector`
- `rotation_delta`
- `gripper_closedness_action`
- `terminate_episode`

이 action은 다음 흐름으로 실제 제어에 반영됩니다.

1. 현재 EE pose에 delta 적용
2. 새로운 target pose 생성
3. IK 계산
4. joint target 생성
5. 실제 로봇 motion 실행

### 4. Post-Grasp Motion

정책이 grasp 요청을 충분히 명확하게 보내거나 코드상 조건이 만족되면 lift 동작을 수행할 수 있습니다.

전체 구조는 다음처럼 역할이 분리됩니다.

- pre-grasp planning
- policy 기반 cartesian 접근
- post-grasp lift

## 주요 코드 구성

### 실행기

- `robotarm_executor/robotarm_executor/chair_grasp_moveit_pi0_lite_policy_external_cartesian_standalone.py`
  - 실제 로봇 실행기
  - 손목/외부 카메라 입력을 받아 정책 추론 수행
  - cartesian action을 IK 기반 joint motion으로 변환

### 학습 코드

- `pi0_lite/pi0_lite_phase_aux_external.py`
  - 손목 이미지 + 외부 이미지 멀티뷰 학습
  - phase auxiliary head 포함
  - flow matching 기반 action trajectory 학습

- `pi0_lite/pi0_lite.py`
  - 단일 이미지 기반 기본 학습 코드
  - `cartesian_delta` / `joint_delta` 지원

### 데이터 수집 및 가공

- `robotarm_executor/robotarm_executor/chair_grasp_moveit_vla_dataset_external.py`
  - 원본 demonstration 데이터 수집
  - 손목/외부 이미지, state, phase 저장

- `pi0_lite/prepare_pi0_lite_phase_aux_external_dataset.py`
  - 원본 수집 데이터를 `pi0_lite_phase_aux_external.py` 학습용 포맷으로 1차 가공

- `pi0_lite/merge_pi0_lite_phase_aux_external_jsonl.py`
  - 여러 episode를 하나의 merged JSONL로 2차 가공

## 데이터 흐름

### 1. 원본 데이터 수집

수집기:
- `chair_grasp_moveit_vla_dataset_external.py`

결과:
- `samples_openvla.jsonl`
- `images/`
- `external_images/`

### 2. 1차 가공

- `prepare_pi0_lite_phase_aux_external_dataset.py`

역할:
- 원본 `samples_openvla.jsonl`을 읽음
- `pi0_lite_phase_aux_external.py` 학습용 포맷으로 변환
- cartesian delta action 생성
- 손목/외부 이미지 경로 유지

### 3. 2차 가공

- `merge_pi0_lite_phase_aux_external_jsonl.py`

역할:
- 여러 episode JSONL을 하나의 merged JSONL로 병합
- `image` / `external_image` 경로 정리
- 학습용 단일 JSONL 생성

### 4. 학습

- `pi0_lite_phase_aux_external.py`

입력:
- `merged_samples_pi0_aux_external_delta.jsonl`

역할:
- 멀티뷰 이미지 + state + instruction 기반 action trajectory 학습
- phase auxiliary loss 함께 학습

### 5. 추론 및 실행

- `chair_grasp_moveit_pi0_lite_policy_external_cartesian_standalone.py`

역할:
- 학습된 checkpoint 로드
- 멀티뷰 입력 기반 cartesian action 예측
- IK 변환 후 실제 로봇 동작 수행

## 학습 방식

이 프로젝트의 정책 학습은 일반적인 다음-action 회귀가 아니라 **flow matching 기반 trajectory 생성 학습**입니다.

모델은 다음 정보를 입력으로 사용합니다.

- 손목 이미지
- 외부 이미지
- instruction
- robot state

그리고 미래 `horizon` 길이의 action trajectory를 한 번에 예측하도록 학습합니다.

추가로 `pi0_lite_phase_aux_external.py`에서는

- phase auxiliary head
- phase loss
- phase별 loss weight

를 함께 사용하여 접근 단계와 grasp 전후 단계의 차이를 더 잘 학습하도록 구성했습니다.

## 왜 멀티뷰가 중요한가

이 프로젝트에서 가장 보고 싶었던 부분은 **외부카메라가 실제 접근 문제를 개선하는지**였습니다.

손목 단일뷰 정책은

- 로봇 말단 기준의 국소 정보는 볼 수 있지만
- 전체 장면과 의자와의 상대 위치를 안정적으로 파악하는 데 한계가 있었습니다.

반면 외부카메라를 추가하면

- 장면 전체 구조
- 로봇팔과 의자의 상대적 위치
- 접근 방향과 정렬

을 함께 볼 수 있기 때문에, 정책이 더 안정적으로 접근 단계를 수행할 수 있을 것으로 기대했습니다.

실제 실험에서도 손목 단일뷰만 사용했을 때보다,

- 더 깊게 내려오는 접근
- 더 일관된 phase progression
- 더 안정적인 접근 반복

이 관찰되었습니다.

즉 멀티뷰 입력은 단순한 입력 확장이 아니라, **접근 문제 자체를 해결하기 위한 핵심 설계 요소**였습니다.

## 현재까지의 결과

### 정성적 결과

- pre-grasp pose까지는 MoveIt2 기반으로 안정적으로 이동
- 최종 접근 구간은 `pi0_lite` cartesian policy가 제어
- 손목 + 외부 이미지 입력을 사용했을 때 접근 품질 개선
- 손목 단일뷰에서 보이던 조기 정지 문제 일부 완화
- 의자 가까이까지 내려오는 동작 품질 향상

### 핵심 개선점

이 프로젝트에서 가장 중요하게 개선한 부분은 다음입니다.

**손목카메라 단일뷰 정책에서 부족했던 의자 접근 성능을, 외부카메라를 추가한 멀티뷰 정책으로 보완할 수 있는지 실제 실행 수준에서 검증했다는 점**

즉

- 단일뷰 접근 한계 확인
- 외부카메라 추가
- 멀티뷰 학습 구성
- 실제 접근 품질 개선 여부 확인

의 흐름이 이 프로젝트의 핵심입니다.

## 아쉬운 점

현재 파이프라인은 외부카메라를 추가한 멀티뷰 입력으로 **접근 품질 자체는 개선**되었지만, 최종적으로 안정적인 grasp 성공까지는 여전히 한계가 있습니다.

가장 크게 드러난 문제는 **close signal이 약하게 나오는 경우가 많았고, 그 결과 실제 집기 동작이 충분히 트리거되지 않는 점**이었습니다.

실행 로그를 보면 정책이

- `approach_far`
- `approach_mid`
- `contact_ready`

등 접근 단계는 비교적 잘 수행하지만,

정작 grasp를 위해 필요한

- `gripper close`
- 명시적인 집기 요청

신호는 충분히 강하게 내지 못하는 경우가 많았습니다.

그 결과

- 의자 가까이까지 접근은 함
- 때로는 접촉하거나 살짝 들어올리는 듯한 motion이 발생함
- 하지만 코드상 명확한 close trigger가 발생하지 않아 안정적인 grasp로 이어지지 못함

현재로서는 이 문제의 중요한 원인 중 하나를 **집기 단계 데이터셋 부족**으로 보고 있습니다.

즉 지금까지의 데이터는

- 접근 단계 데이터는 비교적 많이 개선되었고
- `approach_far → approach_mid → contact_ready` 구간은 더 잘 학습되었지만
- 실제 `close` 직전과 `grasp` 순간 데이터는 상대적으로 부족하거나 다양성이 충분하지 않았습니다.

이 때문에 모델이

- “어떻게 접근해야 하는가”는 점점 더 잘 배우고 있지만
- “언제, 어떤 시점에서 그리퍼를 닫아야 하는가”는 충분히 강하게 학습하지 못한 것으로 보입니다.

즉 현재 한계를 한 줄로 정리하면 다음과 같습니다.

**접근 성능은 멀티뷰 입력으로 개선되었지만, 집기 성공까지 이어지지 못한 주요 원인 중 하나는 close/grasp 단계 데이터셋이 충분하지 않아 정책이 강한 close signal을 학습하지 못한 점입니다.**

## 향후 개선 방향

- `close` 단계 샘플 추가 수집
- `grasp` 직전 장면 데이터 다양성 확대
- 다양한 chair pose / camera view / lighting 조건에서 데이터 추가 수집
- 외부카메라 기반 평가 신호 강화
- close / lift trigger 조건 개선
- inference latency 최적화
- policy 출력과 IK feasibility 사이의 gap 완화
- 접근뿐 아니라 grasp 순간과 lift 이후까지 더 안정적인 후속 행동 설계

## 실행 예시

```bash
ros2 run robotarm_executor chair_grasp_moveit_pi0_lite_policy_external_cartesian_standalone --ros-args \
  -p use_external_camera:=true \
  -p policy_timeout_sec:=300.0 \
  -p policy_max_steps:=50 \
  -p translation_scale:=6.0
