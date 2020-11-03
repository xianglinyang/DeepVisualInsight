/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
import {Injectable} from '@angular/core';
import {Actions, createEffect} from '@ngrx/effects';
import {Store} from '@ngrx/store';
import {tap} from 'rxjs/operators';
import {State} from '../../app_state';
import {alertReported} from '../actions';
import {AlertActionModule} from '../alert_action_module';

/** @typehack */ import * as _typeHackNgrxEffects from '@ngrx/effects/effects';
/** @typehack */ import * as _typeHackStore from '@ngrx/store';
/** @typehack */ import * as _typeHackRxjs from 'rxjs';

@Injectable()
export class AlertEffects {
  constructor(
    private readonly actions$: Actions,
    private readonly store: Store<State>,
    private readonly alertActionModule: AlertActionModule
  ) {}

  /** @export */
  reportRegisteredActionAlerts$ = createEffect(
    () => {
      return this.actions$.pipe(
        tap((action) => {
          const alertInfo = this.alertActionModule.getAlertFromAction(action);
          if (alertInfo) {
            this.store.dispatch(alertReported(alertInfo));
          }
        })
      );
    },
    {dispatch: false}
  );
}
